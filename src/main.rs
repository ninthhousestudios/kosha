use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::EnvFilter;

use kosha::{
    chunk::ChunkConfig,
    config::Config,
    db,
    embed::{Device, EmbedProvider, HttpEmbedder, LocalEmbedder},
    mcp::KoshaServer,
};

/// kosha: document intelligence MCP server.
#[derive(Debug, Parser)]
#[command(name = "kosha", version, about)]
struct Cli {
    /// Device for local embedding: cpu, gpu, auto (default: auto).
    #[arg(long, default_value = "auto", global = true)]
    device: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run as a stdio MCP server (default).
    Serve,
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::from_path(kosha::config::kosha_home().join(".env"));
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();
    let cfg = Config::from_env();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&cfg.log_level)),
        )
        .with_writer(std::io::stderr)
        .init();

    let device = resolve_device(&cli.device)?;

    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Serve => run_serve(cfg, &device).await,
        Commands::List { leaf, collection, format, tags } => {
            run_list(cfg, leaf, collection, format, &tags).await
        }
        Commands::Ingest { paths, recursive, collection, tags } => {
            run_ingest(cfg, &paths, recursive, &collection, &tags, &device).await
        }
    }
}

async fn run_serve(cfg: Config, device: &Device) -> Result<()> {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "kosha starting");

    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    let embedder = build_embedder(&cfg, device).await?;

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

        println!(
            "{:<12} {:<10} {:<12} {:>4} {:>6}  {}",
            "HASH", "FORMAT", "COLLECTION", "SEG", "CHUNKS", "PATH"
        );
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
    device: &Device,
) -> Result<()> {
    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    let embedder = build_embedder(&cfg, device).await?;

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
            let dir_files = kosha::ingest::collect_files(path)
                .with_context(|| format!("walking {p}"))?;
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
                eprintln!(
                    "  {} segments, {} chunks",
                    result.segments, result.chunks
                );
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

fn resolve_device(s: &str) -> Result<Device> {
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

async fn build_embedder(cfg: &Config, device: &Device) -> Result<Arc<dyn EmbedProvider>> {
    match cfg.embed_provider.as_str() {
        "local" => {
            let repo = cfg.model_repo.clone();
            let dim = cfg.embed_dimension;
            let dev = device.clone();
            tracing::info!(%repo, dim, device = ?dev, "loading local embedding model");
            let embedder =
                tokio::task::spawn_blocking(move || LocalEmbedder::load(&repo, dim, &dev))
                    .await
                    .context("join error")?
                    .context("loading local embedder")?;
            Ok(Arc::new(embedder))
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
        other => anyhow::bail!("unknown KOSHA_EMBED_PROVIDER: {other} (expected \"local\" or \"http\")"),
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
