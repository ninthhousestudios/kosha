use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::error::{KoshaError, Result};
use crate::store;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct LeafArgs {
    pub leaf_id: String,
}

#[derive(Debug, Serialize)]
pub struct LeafOutput {
    pub content_hash: String,
    pub source_path: String,
    pub format: String,
    pub title: Option<String>,
    pub collection: String,
    pub tags: Vec<String>,
    pub segment_count: i32,
    pub chunk_count: i64,
    pub status: String,
    pub error: Option<String>,
}

#[tracing::instrument(name = "tool.kosha_leaf", skip(pool))]
pub async fn handle(pool: &PgPool, args: LeafArgs) -> Result<LeafOutput> {
    let leaf = store::get_leaf(pool, &args.leaf_id)
        .await?
        .ok_or_else(|| KoshaError::NotFound {
            tool: "kosha_leaf",
            kind: "leaf",
            next_action: format!(
                "No leaf with id '{}'. Use kosha_leaves to list available leaves.",
                args.leaf_id
            ),
        })?;

    Ok(LeafOutput {
        content_hash: leaf.content_hash,
        source_path: leaf.source_path,
        format: leaf.format,
        title: leaf.title,
        collection: leaf.collection,
        tags: leaf.tags,
        segment_count: leaf.segment_count,
        chunk_count: leaf.chunk_count,
        status: leaf.status,
        error: leaf.error,
    })
}
