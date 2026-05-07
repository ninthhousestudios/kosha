use std::path::Path;

use sqlx::PgPool;

use crate::chunk::{self, ChunkConfig};
use crate::decompose::Decomposer;
use crate::decompose::pdf::PdfDecomposer;
use crate::decompose::plain_text::PlainTextDecomposer;
use crate::embed::EmbedProvider;
use crate::store;

#[derive(Debug)]
pub struct IngestResult {
    pub content_hash: String,
    pub source_path: String,
    pub collection: String,
    pub tags: Vec<String>,
    pub segments: u32,
    pub chunks: u32,
    pub skipped: bool,
}

fn decomposer_for(path: &Path) -> Box<dyn Decomposer> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("pdf") => Box::new(PdfDecomposer),
        _ => Box::new(PlainTextDecomposer),
    }
}

pub async fn ingest_file(
    pool: &PgPool,
    embedder: &dyn EmbedProvider,
    path: &Path,
    chunk_cfg: &ChunkConfig,
    collection: &str,
    tags: &[String],
) -> anyhow::Result<IngestResult> {
    let content = tokio::fs::read(path).await?;
    let source_path = path.to_string_lossy().to_string();

    let content_hash = blake3::hash(&content).to_hex().to_string();
    tracing::info!(%content_hash, %source_path, %collection, "checking leaf");

    match store::leaf_status(pool, &content_hash).await? {
        Some(ref s) if s == "ready" => {
            tracing::info!(%content_hash, "leaf already ingested, skipping");
            return Ok(IngestResult {
                content_hash,
                source_path,
                collection: collection.to_string(),
                tags: tags.to_vec(),
                segments: 0,
                chunks: 0,
                skipped: true,
            });
        }
        Some(ref status) => {
            tracing::info!(%content_hash, %status, "purging partial data for re-ingest");
            store::purge_leaf_children(pool, &content_hash).await?;
        }
        None => {}
    }

    let decomposer = decomposer_for(path);
    let segments = decomposer.decompose(&content, &source_path)?;
    let segment_count = segments.len() as i32;

    store::insert_leaf(
        pool,
        &content_hash,
        &source_path,
        decomposer.format_name(),
        None,
        collection,
        segment_count,
    )
    .await?;

    if !tags.is_empty() {
        store::set_leaf_tags(pool, &content_hash, tags).await?;
    }

    match ingest_segments(pool, embedder, &content_hash, &segments, chunk_cfg).await {
        Ok(total_chunks) => {
            store::mark_leaf_ready(pool, &content_hash).await?;
            tracing::info!(%content_hash, segments = segment_count, chunks = total_chunks, "ingest complete");
            Ok(IngestResult {
                content_hash,
                source_path,
                collection: collection.to_string(),
                tags: tags.to_vec(),
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
    embedder: &dyn EmbedProvider,
    content_hash: &str,
    segments: &[crate::decompose::Segment],
    chunk_cfg: &ChunkConfig,
) -> anyhow::Result<u32> {
    use crate::decompose::SegmentContent;

    let mut total_chunks = 0u32;

    for seg in segments {
        match &seg.content {
            SegmentContent::Text(text) => {
                let segment_id = store::insert_segment(
                    pool,
                    content_hash,
                    seg.index as i32,
                    &seg.label,
                    Some(text),
                )
                .await?;

                let chunks = chunk::chunk_segment(text, &seg.label, chunk_cfg);
                let chunk_texts: Vec<String> =
                    chunks.iter().map(|c| c.content.clone()).collect();
                let embeddings = embedder.embed_batch(chunk_texts).await?;

                for (ch, emb) in chunks.iter().zip(embeddings.iter()) {
                    store::insert_chunk(
                        pool,
                        segment_id,
                        ch.index as i32,
                        content_hash,
                        seg.index as i32,
                        &ch.label,
                        Some(&ch.content),
                        emb,
                        embedder.provider_name(),
                        embedder.model_name(),
                        embedder.dimension() as i32,
                    )
                    .await?;
                    total_chunks += 1;
                }
            }
            SegmentContent::Image(png_data) => {
                let segment_id = store::insert_segment(
                    pool,
                    content_hash,
                    seg.index as i32,
                    &seg.label,
                    None,
                )
                .await?;

                let embeddings = embedder
                    .embed_image_bytes(vec![png_data.clone()])
                    .await?;
                let emb = embeddings.into_iter().next().ok_or_else(|| {
                    anyhow::anyhow!("empty image embedding result for {}", seg.label)
                })?;

                store::insert_chunk(
                    pool,
                    segment_id,
                    0,
                    content_hash,
                    seg.index as i32,
                    &seg.label,
                    None,
                    &emb,
                    embedder.provider_name(),
                    embedder.model_name(),
                    embedder.dimension() as i32,
                )
                .await?;
                total_chunks += 1;
            }
        }
    }

    Ok(total_chunks)
}
