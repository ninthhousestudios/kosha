use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::error::Result;
use crate::store;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct LeavesArgs {
    pub format: Option<String>,
    pub status: Option<String>,
    pub limit: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct LeavesOutput {
    pub leaves: Vec<LeafSummary>,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct LeafSummary {
    pub content_hash: String,
    pub source_path: String,
    pub format: String,
    pub title: Option<String>,
    pub segment_count: i32,
    pub chunk_count: i64,
    pub status: String,
}

#[tracing::instrument(name = "tool.kosha_leaves", skip(pool))]
pub async fn handle(pool: &PgPool, args: LeavesArgs) -> Result<LeavesOutput> {
    let limit = args.limit.unwrap_or(50).clamp(1, 200);

    let records = store::list_leaves(
        pool,
        args.format.as_deref(),
        args.status.as_deref(),
        limit,
    )
    .await?;

    let leaves: Vec<LeafSummary> = records
        .into_iter()
        .map(|r| LeafSummary {
            content_hash: r.content_hash,
            source_path: r.source_path,
            format: r.format,
            title: r.title,
            segment_count: r.segment_count,
            chunk_count: r.chunk_count,
            status: r.status,
        })
        .collect();

    let count = leaves.len();
    Ok(LeavesOutput { leaves, count })
}
