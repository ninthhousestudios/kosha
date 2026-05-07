use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::error::{KoshaError, Result};
use crate::store;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ReadArgs {
    pub leaf_id: String,
    pub segment_index: i32,
    pub chunk_index: Option<i32>,
    pub to_chunk_index: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct ReadOutput {
    pub mode: &'static str,
    pub content: String,
    pub chunks: Vec<ReadChunk>,
}

#[derive(Debug, Serialize)]
pub struct ReadChunk {
    pub chunk_id: String,
    pub chunk_index: i32,
    pub chunk_label: String,
    pub content: String,
}

pub async fn handle(pool: &PgPool, args: ReadArgs) -> Result<ReadOutput> {
    match (args.chunk_index, args.to_chunk_index) {
        (Some(ci), Some(to_ci)) => {
            let records =
                store::read_chunk_range(pool, &args.leaf_id, args.segment_index, ci, to_ci).await?;
            if records.is_empty() {
                return Err(KoshaError::NotFound {
                    tool: "kosha_read",
                    kind: "chunks",
                    next_action: format!(
                        "No chunks found for leaf {} segment {} range {}..{}",
                        args.leaf_id, args.segment_index, ci, to_ci
                    ),
                });
            }
            let content = records
                .iter()
                .map(|r| r.content_text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let chunks = records
                .into_iter()
                .map(|r| ReadChunk {
                    chunk_id: r.id.to_string(),
                    chunk_index: r.chunk_index,
                    chunk_label: r.chunk_label,
                    content: r.content_text,
                })
                .collect();
            Ok(ReadOutput {
                mode: "range",
                content,
                chunks,
            })
        }

        (Some(ci), None) => {
            let record = store::read_chunk(pool, &args.leaf_id, args.segment_index, ci)
                .await?
                .ok_or_else(|| KoshaError::NotFound {
                    tool: "kosha_read",
                    kind: "chunk",
                    next_action: format!(
                        "No chunk found for leaf {} segment {} chunk {}",
                        args.leaf_id, args.segment_index, ci
                    ),
                })?;
            Ok(ReadOutput {
                mode: "chunk",
                content: record.content_text.clone(),
                chunks: vec![ReadChunk {
                    chunk_id: record.id.to_string(),
                    chunk_index: record.chunk_index,
                    chunk_label: record.chunk_label,
                    content: record.content_text,
                }],
            })
        }

        (None, _) => {
            let record = store::read_segment(pool, &args.leaf_id, args.segment_index)
                .await?
                .ok_or_else(|| KoshaError::NotFound {
                    tool: "kosha_read",
                    kind: "segment",
                    next_action: format!(
                        "No segment found for leaf {} segment {}",
                        args.leaf_id, args.segment_index
                    ),
                })?;
            Ok(ReadOutput {
                mode: "segment",
                content: record.content_text,
                chunks: vec![],
            })
        }
    }
}
