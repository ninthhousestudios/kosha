use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::EnvFilter;

use kosha::{
    chunk::ChunkConfig,
    config::Config,
    db,
    embed::{EmbedProvider, HttpEmbedder, LocalEmbedder},
    mcp::KoshaServer,
};

/// kosha: document intelligence MCP server.
#[derive(Debug, Parser)]
#[command(name = "kosha", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run as a stdio MCP server (default).
    Serve,
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

    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Serve => run_serve(cfg).await,
        Commands::Ingest { paths, recursive, collection, tags } => {
            run_ingest(cfg, &paths, recursive, &collection, &tags).await
        }
    }
}

async fn run_serve(cfg: Config) -> Result<()> {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "kosha starting");

    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    let embedder = build_embedder(&cfg).await?;

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

async fn run_ingest(
    cfg: Config,
    paths: &[String],
    recursive: bool,
    collection: &str,
    tags: &[String],
) -> Result<()> {
    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    let embedder = build_embedder(&cfg).await?;

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

async fn build_embedder(cfg: &Config) -> Result<Arc<dyn EmbedProvider>> {
    match cfg.embed_provider.as_str() {
        "local" => {
            let repo = cfg.model_repo.clone();
            let dim = cfg.embed_dimension;
            tracing::info!(%repo, dim, "loading local embedding model");
            let embedder = tokio::task::spawn_blocking(move || LocalEmbedder::load(&repo, dim))
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
