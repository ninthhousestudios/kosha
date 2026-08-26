use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::error::{KoshaError, Result};
use crate::store;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DocumentArgs {
    /// Stable source id — a search hit's citation `source_path` reduced to its
    /// file stem (e.g. `dhata-rakshasa`). Resolves to exactly one document.
    pub source_id: String,
    /// Optional collection scope (names). Omit to resolve across all collections.
    pub collections: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct DocumentOutput {
    pub source_id: String,
    pub title: Option<String>,
    pub format: String,
    pub content: String,
}

/// Fetch a whole document by its stable `source_id`. Where `kosha_read` returns
/// one segment/chunk keyed by the internal `leaf_id` hash, this reassembles the
/// entire document (all segments in order) from a caller-facing id — the piece a
/// backend needs to expand a search hit to its full source.
#[tracing::instrument(name = "tool.kosha_document", skip(pool))]
pub async fn handle(pool: &PgPool, args: DocumentArgs) -> Result<DocumentOutput> {
    let doc = store::read_document_by_source_id(pool, &args.source_id, args.collections.as_deref())
        .await?
        .ok_or_else(|| KoshaError::NotFound {
            tool: "kosha_document",
            kind: "document",
            next_action: format!(
                "No document with source_id '{}'. Use kosha_search to discover valid source ids.",
                args.source_id
            ),
        })?;

    Ok(DocumentOutput {
        source_id: args.source_id,
        title: doc.title,
        format: doc.format,
        content: doc.content,
    })
}
