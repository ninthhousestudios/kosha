use std::path::Path;
use std::sync::Arc;

use sqlx::PgPool;

use crate::chunk::{self, ChunkConfig};
use crate::decompose::plain_text::PlainTextDecomposer;
use crate::decompose::Decomposer;
use crate::embed::Embedder;
use crate::store;

#[derive(Debug)]
pub struct IngestResult {
    pub content_hash: String,
    pub source_path: String,
    pub segments: u32,
    pub chunks: u32,
    pub skipped: bool,
}

pub async fn ingest_file(
    pool: &PgPool,
    embedder: &Arc<Embedder>,
    path: &Path,
    chunk_cfg: &ChunkConfig,
) -> anyhow::Result<IngestResult> {
    let content = tokio::fs::read_to_string(path).await?;
    let source_path = path.to_string_lossy().to_string();

    let content_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    tracing::info!(%content_hash, %source_path, "checking leaf");

    if store::leaf_exists(pool, &content_hash).await? {
        tracing::info!(%content_hash, "leaf already exists, skipping");
        return Ok(IngestResult {
            content_hash,
            source_path,
            segments: 0,
            chunks: 0,
            skipped: true,
        });
    }

    let decomposer = PlainTextDecomposer;
    let segments = decomposer.decompose(&content, &source_path);
    let segment_count = segments.len() as i32;

    store::insert_leaf(
        pool,
        &content_hash,
        &source_path,
        decomposer.format_name(),
        None,
        segment_count,
    )
    .await?;

    match ingest_segments(pool, embedder, &content_hash, &segments, chunk_cfg).await {
        Ok(total_chunks) => {
            store::mark_leaf_ready(pool, &content_hash).await?;
            tracing::info!(%content_hash, segments = segment_count, chunks = total_chunks, "ingest complete");
            Ok(IngestResult {
                content_hash,
                source_path,
                segments: segment_count as u32,
                chunks: total_chunks,
                skipped: false,
            })
        }
        Err(e) => {
            let _ = store::mark_leaf_error(pool, &content_hash, &e.to_string()).await;
            Err(e)
        }
    }
}

async fn ingest_segments(
    pool: &PgPool,
    embedder: &Arc<Embedder>,
    content_hash: &str,
    segments: &[crate::decompose::Segment],
    chunk_cfg: &ChunkConfig,
) -> anyhow::Result<u32> {
    let mut total_chunks = 0u32;

    for seg in segments {
        let segment_id = store::insert_segment(
            pool,
            content_hash,
            seg.index as i32,
            &seg.label,
            &seg.content,
        )
        .await?;

        let chunks = chunk::chunk_segment(&seg.content, &seg.label, chunk_cfg);

        let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let embeddings = embedder.embed_batch_async(chunk_texts).await?;

        for (ch, emb) in chunks.iter().zip(embeddings.iter()) {
            store::insert_chunk(
                pool,
                segment_id,
                ch.index as i32,
                content_hash,
                seg.index as i32,
                &ch.label,
                &ch.content,
                emb,
            )
            .await?;
            total_chunks += 1;
        }
    }

    Ok(total_chunks)
}
