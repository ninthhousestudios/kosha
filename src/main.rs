use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::EnvFilter;

use kosha::{
    chunk::ChunkConfig,
    config::Config,
    db,
    embed::{EmbedProvider, HttpEmbedder},
    mcp::KoshaServer,
    tools::SearchArgs,
};

/// kosha: document intelligence MCP server.
#[derive(Debug, Parser)]
#[command(name = "kosha", version, about)]
struct Cli {
    /// Device for local embedding: cpu, gpu, auto (default: auto).
    #[arg(long, default_value = "auto", global = true)]
    device: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run as a server: stdio MCP (default), or a thin HTTP search surface with --http.
    Serve {
        /// Serve the thin HTTP surface (POST /search, GET|POST /health) instead of stdio MCP.
        #[arg(long)]
        http: bool,
    },
    /// List ingested documents, or show outline for a specific leaf.
    List {
        /// Content hash (prefix ok) to show outline for. Omit to list all leaves.
        leaf: Option<String>,
        /// Filter by collection.
        #[arg(long)]
        collection: Option<String>,
        /// Filter by format (e.g. pdf, epub, markdown).
        #[arg(long)]
        format: Option<String>,
        /// Filter by tag (repeatable).
        #[arg(long = "tag", num_args = 1)]
        tags: Vec<String>,
    },
    /// Ingest documents into the database.
    Ingest {
        /// Files or directories to ingest.
        #[arg(required = true, num_args = 1..)]
        paths: Vec<String>,
        /// Recurse into directories.
        #[arg(short, long)]
        recursive: bool,
        /// Collection to assign leaves to.
        #[arg(long, default_value = "default")]
        collection: String,
        /// Tags to attach to leaves (repeatable).
        #[arg(long = "tag", num_args = 1)]
        tags: Vec<String>,
    },
    /// Semantic search over ingested documents.
    Search {
        /// Natural language query.
        query: String,
        /// Filter by collection (repeatable).
        #[arg(long = "collection", num_args = 1)]
        collections: Vec<String>,
        /// Filter by tag (repeatable).
        #[arg(long = "tag", num_args = 1)]
        tags: Vec<String>,
        /// Max results to return.
        #[arg(long, default_value = "5")]
        limit: i64,
        /// Output raw JSON instead of formatted text.
        #[arg(long)]
        json: bool,
    },
    /// List embedding models supported by the ONNX provider.
    Models,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::from_path(kosha::config::kosha_home().join(".env"));
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();

    // `models` lists static model metadata; it needs neither the DB nor config.
    if matches!(cli.command, Commands::Models) {
        return run_models();
    }

    let cfg = Config::from_env();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cfg.log_level)),
        )
        .with_writer(std::io::stderr)
        .init();

    // The `--device` string is resolved to a concrete candle device lazily,
    // inside the (candle-gated) local-embedder path — the onnx/http providers
    // ignore it, and the candle `Device` type is absent from onnx-only builds.
    let Cli { device, command } = cli;

    match command {
        Commands::Serve { http } => {
            if http {
                run_serve_http(cfg, &device).await
            } else {
                run_serve(cfg, &device).await
            }
        }
        Commands::List {
            leaf,
            collection,
            format,
            tags,
        } => run_list(cfg, leaf, collection, format, &tags).await,
        Commands::Ingest {
            paths,
            recursive,
            collection,
            tags,
        } => run_ingest(cfg, &paths, recursive, &collection, &tags, &device).await,
        Commands::Search {
            query,
            collections,
            tags,
            limit,
            json,
        } => run_search(cfg, query, collections, tags, limit, json, &device).await,
        Commands::Models => run_models(),
    }
}

/// Print the embedding models supported by the ONNX provider, with their
/// output dimensions. The printed name is the value to pass as KOSHA_EMBED_MODEL.
fn run_models() -> Result<()> {
    let mut models = fastembed::TextEmbedding::list_supported_models();
    models.sort_by_key(|m| m.model.to_string());
    #[expect(
        clippy::print_literal,
        reason = "header literals kept as args so this format string stays identical to the data-row one below"
    )]
    {
        println!(
            "{:<28} {:>5}  {}",
            "MODEL (KOSHA_EMBED_MODEL)", "DIM", "DESCRIPTION"
        );
    }
    for m in &models {
        println!(
            "{:<28} {:>5}  {}",
            m.model.to_string(),
            m.dim,
            m.description
        );
    }
    Ok(())
}

async fn run_serve(cfg: Config, device: &str) -> Result<()> {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "kosha starting");

    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    let embedder = build_embedder(&cfg, device).await?;
    db::ensure_embedding_dim(&pool, embedder.dimension())
        .await
        .context("reconciling embedding dimension")?;

    let server = KoshaServer::new(pool, embedder);
    let service = server.serve(stdio()).await.context("starting MCP server")?;

    tokio::select! {
        res = service.waiting() => {
            res.context("MCP server exited")?;
        }
        _ = shutdown_signal() => {
            tracing::info!("shutdown signal received");
        }
    }

    Ok(())
}

/// Serve the thin HTTP surface (`POST /search`, `GET|POST /health`) instead of
/// stdio MCP. Same pool/embedder setup as `run_serve`; the backend calls this
/// from behind its own op surface. Listen address comes from `KOSHA_HTTP_ADDR`/
/// `KOSHA_HTTP_PORT` (default `0.0.0.0:3400`).
async fn run_serve_http(cfg: Config, device: &str) -> Result<()> {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "kosha HTTP starting");

    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    let embedder = build_embedder(&cfg, device).await?;
    db::ensure_embedding_dim(&pool, embedder.dimension())
        .await
        .context("reconciling embedding dimension")?;

    let addr: SocketAddr = format!("{}:{}", cfg.http_addr, cfg.http_port)
        .parse()
        .with_context(|| {
            format!(
                "invalid HTTP listen address {}:{}",
                cfg.http_addr, cfg.http_port
            )
        })?;

    tokio::select! {
        res = kosha::serve_http::serve(pool, embedder, addr) => {
            res.context("HTTP server exited")?;
        }
        _ = shutdown_signal() => {
            tracing::info!("shutdown signal received");
        }
    }

    Ok(())
}

async fn run_list(
    cfg: Config,
    leaf: Option<String>,
    collection: Option<String>,
    format: Option<String>,
    tags: &[String],
) -> Result<()> {
    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    if let Some(prefix) = leaf {
        let full_hash = kosha::store::resolve_hash_prefix(&pool, &prefix)
            .await?
            .with_context(|| format!("no leaf matches hash prefix '{prefix}'"))?;
        let leaf_info = kosha::store::get_leaf(&pool, &full_hash).await?;
        if let Some(info) = &leaf_info {
            println!("{} ({})", info.source_path, info.format);
            println!(
                "{} segments, {} chunks, collection: {}",
                info.segment_count, info.chunk_count, info.collection
            );
            if !info.tags.is_empty() {
                println!("tags: {}", info.tags.join(", "));
            }
            println!();
        }
        let outline = kosha::store::leaf_outline(&pool, &full_hash).await?;
        for entry in &outline {
            println!("  {:>3}  {}", entry.segment_index, entry.segment_label);
        }
    } else {
        let colls: Option<Vec<String>> = collection.map(|c| vec![c]);
        let tag_vec: Option<&[String]> = if tags.is_empty() { None } else { Some(tags) };
        let leaves = kosha::store::list_leaves(
            &pool,
            format.as_deref(),
            Some("ready"),
            colls.as_deref(),
            tag_vec,
            500,
        )
        .await?;

        if leaves.is_empty() {
            println!("No documents ingested.");
            return Ok(());
        }

        #[expect(
            clippy::print_literal,
            reason = "header literals kept as args so this format string stays identical to the data-row one below"
        )]
        {
            println!(
                "{:<12} {:<10} {:<12} {:>4} {:>6}  {}",
                "HASH", "FORMAT", "COLLECTION", "SEG", "CHUNKS", "PATH"
            );
        }
        for leaf in &leaves {
            let hash_short = if leaf.content_hash.len() > 10 {
                &leaf.content_hash[..10]
            } else {
                &leaf.content_hash
            };
            println!(
                "{:<12} {:<10} {:<12} {:>4} {:>6}  {}",
                hash_short,
                leaf.format,
                leaf.collection,
                leaf.segment_count,
                leaf.chunk_count,
                leaf.source_path,
            );
        }
        println!("\n{} document(s)", leaves.len());
    }

    Ok(())
}

async fn run_ingest(
    cfg: Config,
    paths: &[String],
    recursive: bool,
    collection: &str,
    tags: &[String],
    device: &str,
) -> Result<()> {
    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    let embedder = build_embedder(&cfg, device).await?;
    db::ensure_embedding_dim(&pool, embedder.dimension())
        .await
        .context("reconciling embedding dimension")?;

    let chunk_cfg = ChunkConfig {
        target_tokens: cfg.chunk_target_tokens,
        tolerance_tokens: cfg.chunk_tolerance_tokens,
        overlap_tokens: cfg.chunk_overlap_tokens,
    };

    let mut files = Vec::new();
    for p in paths {
        let path = std::path::Path::new(p);
        if path.is_dir() {
            if !recursive {
                anyhow::bail!(
                    "{} is a directory; use -r/--recursive to ingest directories",
                    p
                );
            }
            let dir_files =
                kosha::ingest::collect_files(path).with_context(|| format!("walking {p}"))?;
            files.extend(dir_files);
        } else {
            files.push(path.to_path_buf());
        }
    }

    let total = files.len();
    let mut ingested = 0u32;
    let mut skipped = 0u32;
    let mut errors = 0u32;

    for (i, file_path) in files.iter().enumerate() {
        let display = file_path.display();
        eprintln!("[{}/{}] {}", i + 1, total, display);

        match kosha::ingest::ingest_file(
            &pool,
            embedder.as_ref(),
            file_path,
            &chunk_cfg,
            collection,
            tags,
        )
        .await
        {
            Ok(result) if result.skipped => {
                eprintln!("  skipped (already ingested)");
                skipped += 1;
            }
            Ok(result) => {
                eprintln!("  {} segments, {} chunks", result.segments, result.chunks);
                ingested += 1;
            }
            Err(e) => {
                eprintln!("  error: {e:#}");
                errors += 1;
            }
        }
    }

    eprintln!(
        "\nkosha: {ingested} ingested, {skipped} skipped, {errors} errors (of {total} files)"
    );

    if errors > 0 {
        anyhow::bail!("{errors} file(s) failed to ingest");
    }

    Ok(())
}

async fn run_search(
    cfg: Config,
    query: String,
    collections: Vec<String>,
    tags: Vec<String>,
    limit: i64,
    json: bool,
    device: &str,
) -> Result<()> {
    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    let embedder = build_embedder(&cfg, device).await?;
    db::ensure_embedding_dim(&pool, embedder.dimension())
        .await
        .context("reconciling embedding dimension")?;

    let args = SearchArgs {
        query,
        collections: if collections.is_empty() {
            None
        } else {
            Some(collections)
        },
        tags: if tags.is_empty() { None } else { Some(tags) },
        limit: Some(limit),
    };

    let output = kosha::tools::search::handle(&pool, embedder.as_ref(), args)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if json {
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    if output.results.is_empty() {
        println!("No results for: {}", output.query);
        return Ok(());
    }

    println!("{} result(s) for: {}\n", output.count, output.query);
    for (i, hit) in output.results.iter().enumerate() {
        println!(
            "{}. [score: {:.4}] {}",
            i + 1,
            hit.score,
            hit.citation.source_path
        );
        println!(
            "   {} (chunk {})",
            hit.citation.chunk_label, hit.citation.chunk_index
        );
        let preview: String = hit.content.chars().take(200).collect();
        let ellipsis = if hit.content.len() > 200 { "..." } else { "" };
        println!("   {}{}\n", preview, ellipsis);
    }

    Ok(())
}

/// Resolve the `--device` string to a concrete candle device. Only the local
/// (candle) embedder consumes a device, so this and the `Device` type exist
/// only in candle-backend builds.
#[cfg(feature = "candle-backend")]
fn resolve_device(s: &str) -> Result<kosha::embed::Device> {
    use kosha::embed::Device;
    match s {
        "cpu" => Ok(Device::Cpu),
        "gpu" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).context("failed to initialize CUDA device")
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("--device gpu requires kosha built with --features cuda")
            }
        }
        "auto" => {
            #[cfg(feature = "cuda")]
            {
                match Device::new_cuda(0) {
                    Ok(dev) => {
                        tracing::info!("auto-detected CUDA device");
                        Ok(dev)
                    }
                    Err(_) => {
                        tracing::info!("no CUDA device available, falling back to CPU");
                        Ok(Device::Cpu)
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                Ok(Device::Cpu)
            }
        }
        other => anyhow::bail!("unknown --device value: {other} (expected cpu, gpu, or auto)"),
    }
}

async fn build_embedder(cfg: &Config, device: &str) -> Result<Arc<dyn EmbedProvider>> {
    match cfg.embed_provider.as_str() {
        "local" => {
            #[cfg(feature = "candle-backend")]
            {
                let repo = cfg.model_repo.clone();
                let dim = cfg.embed_dimension;
                let dev = resolve_device(device)?;
                tracing::info!(%repo, dim, device = ?dev, "loading local embedding model");
                let embedder = tokio::task::spawn_blocking(move || {
                    kosha::embed::LocalEmbedder::load(&repo, dim, &dev)
                })
                .await
                .context("join error")?
                .context("loading local embedder")?;
                Ok(Arc::new(embedder))
            }
            #[cfg(not(feature = "candle-backend"))]
            {
                let _ = device;
                anyhow::bail!(
                    "KOSHA_EMBED_PROVIDER=local requires kosha built with the candle-backend feature (the default build)"
                )
            }
        }
        "http" => {
            let url = cfg
                .embed_url
                .as_ref()
                .context("KOSHA_EMBED_URL required when KOSHA_EMBED_PROVIDER=http")?
                .clone();
            let model = cfg
                .embed_model
                .as_ref()
                .context("KOSHA_EMBED_MODEL required when KOSHA_EMBED_PROVIDER=http")?
                .clone();
            tracing::info!(%url, %model, dim = cfg.embed_dimension, "using HTTP embedding provider");
            let embedder = HttpEmbedder::new(
                url,
                model,
                cfg.embed_dimension,
                cfg.embed_api_key.clone(),
                cfg.embed_batch_size,
            );
            Ok(Arc::new(embedder))
        }
        "onnx" => {
            #[cfg(feature = "onnx")]
            {
                let model = cfg
                    .embed_model
                    .as_ref()
                    .context("KOSHA_EMBED_MODEL required when KOSHA_EMBED_PROVIDER=onnx")?
                    .clone();
                let batch = cfg.embed_batch_size;
                let dim_override = cfg.embed_dimension_override;
                tracing::info!(%model, ?dim_override, "loading ONNX embedding model");
                let embedder = tokio::task::spawn_blocking(move || {
                    kosha::embed::OnnxEmbedder::load(&model, batch, dim_override)
                })
                .await
                .context("join error")?
                .context("loading ONNX embedder")?;
                Ok(Arc::new(embedder))
            }
            #[cfg(not(feature = "onnx"))]
            {
                anyhow::bail!("KOSHA_EMBED_PROVIDER=onnx requires kosha built with --features onnx")
            }
        }
        other => anyhow::bail!(
            "unknown KOSHA_EMBED_PROVIDER: {other} (expected \"local\", \"http\", or \"onnx\")"
        ),
    }
}

#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = int.recv() => {}
        _ = term.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
