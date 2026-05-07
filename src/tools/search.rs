use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::embed::EmbedProvider;
use crate::error::{KoshaError, Result};
use crate::store;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchArgs {
    pub query: String,
    pub collections: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub limit: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct SearchOutput {
    pub results: Vec<SearchHit>,
    pub query: String,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub chunk_id: String,
    pub score: f64,
    pub content: String,
    pub citation: Citation,
}

#[derive(Debug, Serialize)]
pub struct Citation {
    pub leaf_id: String,
    pub source_path: String,
    pub segment_index: i32,
    pub chunk_index: i32,
    pub chunk_label: String,
}

pub async fn handle(
    pool: &PgPool,
    embedder: &dyn EmbedProvider,
    args: SearchArgs,
) -> Result<SearchOutput> {
    let limit = args.limit.unwrap_or(5).clamp(1, 20);

    let query_embedding = embedder
        .embed_one(args.query.clone())
        .await
        .map_err(|e| KoshaError::Embed(e.to_string()))?;

    let results = store::search(
        pool,
        &query_embedding,
        args.collections.as_deref(),
        args.tags.as_deref(),
        limit,
    )
    .await?;

    let hits: Vec<SearchHit> = results
        .into_iter()
        .map(|r| SearchHit {
            chunk_id: r.chunk_id.to_string(),
            score: r.score,
            content: r.content_text,
            citation: Citation {
                leaf_id: r.leaf_id,
                source_path: r.source_path,
                segment_index: r.segment_index,
                chunk_index: r.chunk_index,
                chunk_label: r.chunk_label,
            },
        })
        .collect();

    let count = hits.len();
    Ok(SearchOutput {
        results: hits,
        query: args.query,
        count,
    })
}
