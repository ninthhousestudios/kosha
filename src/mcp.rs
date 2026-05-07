use std::sync::Arc;

use rmcp::{
    ErrorData, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use sqlx::PgPool;

use crate::embed::EmbedProvider;
use crate::error::KoshaError;
use crate::tools;

#[derive(Clone)]
pub struct KoshaServer {
    pool: PgPool,
    embedder: Arc<dyn EmbedProvider>,
    tool_router: ToolRouter<Self>,
}

impl KoshaServer {
    pub fn new(pool: PgPool, embedder: Arc<dyn EmbedProvider>) -> Self {
        Self {
            pool,
            embedder,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router(router = tool_router)]
impl KoshaServer {
    #[tool(description = "Health check. Verifies DB connectivity and reports server version.")]
    pub async fn kosha_health(
        &self,
        Parameters(_args): Parameters<tools::HealthArgs>,
    ) -> Result<String, ErrorData> {
        let out = tools::health::handle(&self.pool)
            .await
            .map_err(kosha_to_rmcp)?;
        serde_json::to_string(&out).map_err(json_to_rmcp)
    }

    #[tool(
        description = "Semantic search over ingested documents. Returns ranked chunk results with content snippets and citations (leaf_id, source_path, segment/chunk indices). Use citations with kosha_read for surrounding context."
    )]
    pub async fn kosha_search(
        &self,
        Parameters(args): Parameters<tools::SearchArgs>,
    ) -> Result<String, ErrorData> {
        let out = tools::search::handle(&self.pool, &*self.embedder, args)
            .await
            .map_err(kosha_to_rmcp)?;
        serde_json::to_string(&out).map_err(json_to_rmcp)
    }

    #[tool(
        description = "Read document content by citation. Provide leaf_id + segment_index to read a full segment. Add chunk_index for a single chunk. Add to_chunk_index for a chunk range. Use after kosha_search to expand context around a hit."
    )]
    pub async fn kosha_read(
        &self,
        Parameters(args): Parameters<tools::ReadArgs>,
    ) -> Result<String, ErrorData> {
        let out = tools::read::handle(&self.pool, args)
            .await
            .map_err(kosha_to_rmcp)?;
        serde_json::to_string(&out).map_err(json_to_rmcp)
    }

    #[tool(
        description = "Get metadata for a single ingested leaf (document). Returns format, segment/chunk counts, status, source path, and content hash."
    )]
    pub async fn kosha_leaf(
        &self,
        Parameters(args): Parameters<tools::LeafArgs>,
    ) -> Result<String, ErrorData> {
        let out = tools::leaf::handle(&self.pool, args)
            .await
            .map_err(kosha_to_rmcp)?;
        serde_json::to_string(&out).map_err(json_to_rmcp)
    }

    #[tool(
        description = "List ingested leaves (documents). Optional filters: format (e.g. 'plain_text'), status ('ready', 'processing', 'error'). Returns summary metadata for each leaf."
    )]
    pub async fn kosha_leaves(
        &self,
        Parameters(args): Parameters<tools::LeavesArgs>,
    ) -> Result<String, ErrorData> {
        let out = tools::leaves::handle(&self.pool, args)
            .await
            .map_err(kosha_to_rmcp)?;
        serde_json::to_string(&out).map_err(json_to_rmcp)
    }

    #[tool(
        description = "Get the segment outline (table of contents) for a leaf. Returns segment labels in order — useful for navigating a document before reading specific segments."
    )]
    pub async fn kosha_outline(
        &self,
        Parameters(args): Parameters<tools::OutlineArgs>,
    ) -> Result<String, ErrorData> {
        let out = tools::outline::handle(&self.pool, args)
            .await
            .map_err(kosha_to_rmcp)?;
        serde_json::to_string(&out).map_err(json_to_rmcp)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for KoshaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "kosha — document intelligence layer. Semantic search and citation over ingested documents.",
        )
    }
}

fn kosha_to_rmcp(e: KoshaError) -> ErrorData {
    let code = e.code();
    let message = e.to_string();
    let data = serde_json::to_value(e.data()).ok();
    if code == -32602 {
        ErrorData::invalid_params(message, data)
    } else {
        ErrorData::internal_error(message, data)
    }
}

fn json_to_rmcp(e: serde_json::Error) -> ErrorData {
    ErrorData::internal_error(
        format!("failed to serialize response: {e}"),
        Some(serde_json::json!({
            "tool": "server",
            "constraint": "response serializes to JSON",
            "next_action": "Report this as a bug; include server logs.",
        })),
    )
}
