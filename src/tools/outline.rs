use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::error::{KoshaError, Result};
use crate::store;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct OutlineArgs {
    pub leaf_id: String,
}

#[derive(Debug, Serialize)]
pub struct OutlineOutput {
    pub leaf_id: String,
    pub segments: Vec<SegmentEntry>,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct SegmentEntry {
    pub segment_index: i32,
    pub segment_label: String,
}

#[tracing::instrument(name = "tool.kosha_outline", skip(pool))]
pub async fn handle(pool: &PgPool, args: OutlineArgs) -> Result<OutlineOutput> {
    // Verify the leaf exists first
    let _leaf = store::get_leaf(pool, &args.leaf_id)
        .await?
        .ok_or_else(|| KoshaError::NotFound {
            tool: "kosha_outline",
            kind: "leaf",
            next_action: format!(
                "No leaf with id '{}'. Use kosha_leaves to list available leaves.",
                args.leaf_id
            ),
        })?;

    let entries = store::leaf_outline(pool, &args.leaf_id).await?;

    let segments: Vec<SegmentEntry> = entries
        .into_iter()
        .map(|e| SegmentEntry {
            segment_index: e.segment_index,
            segment_label: e.segment_label,
        })
        .collect();

    let count = segments.len();
    Ok(OutlineOutput {
        leaf_id: args.leaf_id,
        segments,
        count,
    })
}
