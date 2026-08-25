//! Thin HTTP surface over kosha's semantic search.
//!
//! kosha's primary interface is stdio MCP (`src/mcp.rs`). This module adds a
//! deliberately minimal plain-JSON HTTP server (`POST /search`, `GET|POST
//! /health`) so a backend service can call search from behind its own op
//! surface without speaking MCP. It reuses the exact same
//! `tools::search::handle` the MCP tool does — no query logic lives here.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use sqlx::PgPool;
use tokio::net::TcpListener;

use crate::embed::EmbedProvider;
use crate::error::KoshaError;
use crate::tools::{self, SearchArgs};

#[derive(Clone)]
struct HttpState {
    pool: PgPool,
    embedder: Arc<dyn EmbedProvider>,
}

/// Bind `addr` and serve the HTTP surface until the future is dropped or errors.
///
/// Routes: `POST /search` and `GET|POST /health`. The caller (main) races this
/// against the process shutdown signal, so no graceful-shutdown wiring lives
/// here — dropping the future closes the listener.
pub async fn serve(
    pool: PgPool,
    embedder: Arc<dyn EmbedProvider>,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health).post(health))
        .route("/search", post(search))
        .with_state(HttpState { pool, embedder });

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding kosha HTTP listener on {addr}"))?;
    tracing::info!(%addr, "kosha HTTP search server listening");
    axum::serve(listener, app)
        .await
        .context("kosha HTTP server error")?;
    Ok(())
}

/// `GET|POST /health` — reuses the MCP health tool. Returns 200 when the DB is
/// reachable, 503 when degraded, so a container healthcheck sees a real signal.
async fn health(State(state): State<HttpState>) -> Response {
    match tools::health::handle(&state.pool).await {
        Ok(out) => {
            let code = if out.db_connected {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            (code, Json(out)).into_response()
        }
        Err(e) => ApiError(e).into_response(),
    }
}

/// `POST /search` — semantic search. Body is `SearchArgs` (query + optional
/// collections/tags/limit); the response is the same `SearchOutput` the MCP
/// tool returns (ranked hits with citations). The embedding provider must match
/// the one the corpus was ingested with (onnx `NomicEmbedTextV15`, 768-dim) or
/// cosine ranking is meaningless.
async fn search(State(state): State<HttpState>, Json(args): Json<SearchArgs>) -> Response {
    match tools::search::handle(&state.pool, &*state.embedder, args).await {
        Ok(out) => Json(out).into_response(),
        Err(e) => ApiError(e).into_response(),
    }
}

/// Wraps a `KoshaError` for HTTP, reusing its JSON-RPC-style code split:
/// invalid-params (-32602) -> 400, everything else -> 500. The body is the
/// structured `ErrorData` (tool / constraint / next_action).
struct ApiError(KoshaError);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = if self.0.code() == -32602 {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (status, Json(self.0.data())).into_response()
    }
}
