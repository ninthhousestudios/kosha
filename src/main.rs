use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::EnvFilter;

use kosha::{chunk::ChunkConfig, config::Config, db, embed::Embedder, mcp::KoshaServer};

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
        /// Path to a file or directory to ingest.
        path: String,
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
        Commands::Ingest { path } => run_ingest(cfg, path).await,
    }
}

async fn run_serve(cfg: Config) -> Result<()> {
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "kosha starting");

    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    tracing::info!(repo = %cfg.model_repo, "loading embedding model");
    let embedder = load_embedder(&cfg.model_repo).await?;
    let embedder = Arc::new(embedder);

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

async fn run_ingest(cfg: Config, path: String) -> Result<()> {
    let pool = db::create_pool(&cfg).await.context("creating DB pool")?;
    db::run_migrations(&pool)
        .await
        .context("running migrations")?;

    tracing::info!(repo = %cfg.model_repo, "loading embedding model");
    let embedder = load_embedder(&cfg.model_repo).await?;
    let embedder = Arc::new(embedder);

    let chunk_cfg = ChunkConfig {
        max_tokens: cfg.chunk_max_tokens,
        overlap_tokens: cfg.chunk_overlap_tokens,
    };

    let file_path = std::path::Path::new(&path);
    let result = kosha::ingest::ingest_file(&pool, &embedder, file_path, &chunk_cfg).await?;

    if result.skipped {
        eprintln!("kosha: already ingested (hash {})", result.content_hash);
    } else {
        eprintln!(
            "kosha: ingested {} ({} segments, {} chunks, hash {})",
            result.source_path, result.segments, result.chunks, result.content_hash
        );
    }

    Ok(())
}

async fn load_embedder(repo: &str) -> Result<Embedder> {
    let repo = repo.to_string();
    tokio::task::spawn_blocking(move || Embedder::load(&repo))
        .await
        .context("join error")?
        .context("loading embedder")
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
