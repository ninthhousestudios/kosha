use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::EnvFilter;

use kosha::{config::Config, db, mcp::KoshaServer};

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
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cfg.log_level)),
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
    db::run_migrations(&pool).await.context("running migrations")?;

    let server = KoshaServer::new(pool);
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

async fn run_ingest(_cfg: Config, path: String) -> Result<()> {
    tracing::info!(%path, "ingest not yet implemented");
    eprintln!("kosha ingest: not yet implemented (see kosha/8 for embedding prerequisites)");
    Ok(())
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
