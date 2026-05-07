use half::f16;
use pgvector::HalfVector;
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::Result;

fn f32_to_halfvec(v: &[f32]) -> HalfVector {
    HalfVector::from(v.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>())
}

// ── Insert operations ──

pub async fn leaf_status(pool: &PgPool, content_hash: &str) -> Result<Option<String>> {
    let row = sqlx::query_scalar::<_, String>("SELECT status FROM leaves WHERE content_hash = $1")
        .bind(content_hash)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

pub async fn insert_leaf(
    pool: &PgPool,
    content_hash: &str,
    source_path: &str,
    format: &str,
    title: Option<&str>,
    segment_count: i32,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO leaves (content_hash, source_path, format, title, segment_count, status)
         VALUES ($1, $2, $3, $4, $5, 'processing')
         ON CONFLICT (content_hash) DO UPDATE SET
           source_path = EXCLUDED.source_path,
           segment_count = EXCLUDED.segment_count,
           status = 'processing',
           error = NULL,
           updated_at = now()",
    )
    .bind(content_hash)
    .bind(source_path)
    .bind(format)
    .bind(title)
    .bind(segment_count)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn mark_leaf_ready(pool: &PgPool, content_hash: &str) -> Result<()> {
    sqlx::query("UPDATE leaves SET status = 'ready', updated_at = now() WHERE content_hash = $1")
        .bind(content_hash)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn mark_leaf_error(pool: &PgPool, content_hash: &str, error: &str) -> Result<()> {
    sqlx::query(
        "UPDATE leaves SET status = 'error', error = $2, updated_at = now() WHERE content_hash = $1",
    )
    .bind(content_hash)
    .bind(error)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn purge_leaf_children(pool: &PgPool, content_hash: &str) -> Result<()> {
    sqlx::query("DELETE FROM chunks WHERE leaf_id = $1")
        .bind(content_hash)
        .execute(pool)
        .await?;
    sqlx::query("DELETE FROM segments WHERE leaf_id = $1")
        .bind(content_hash)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn insert_segment(
    pool: &PgPool,
    leaf_id: &str,
    segment_index: i32,
    segment_label: &str,
    content_text: &str,
) -> Result<Uuid> {
    let id = Uuid::now_v7();
    let actual_id = sqlx::query_scalar::<_, Uuid>(
        "INSERT INTO segments (id, leaf_id, segment_index, segment_label, content_text)
         VALUES ($1, $2, $3, $4, $5)
         ON CONFLICT (leaf_id, segment_index) DO UPDATE SET
           segment_label = EXCLUDED.segment_label,
           content_text = EXCLUDED.content_text
         RETURNING id",
    )
    .bind(id)
    .bind(leaf_id)
    .bind(segment_index)
    .bind(segment_label)
    .bind(content_text)
    .fetch_one(pool)
    .await?;
    Ok(actual_id)
}

pub async fn insert_chunk(
    pool: &PgPool,
    segment_id: Uuid,
    chunk_index: i32,
    leaf_id: &str,
    segment_index: i32,
    chunk_label: &str,
    content_text: &str,
    embedding: &[f32],
    embed_provider: &str,
    embed_model: &str,
    embed_dimension: i32,
) -> Result<()> {
    let id = Uuid::now_v7();
    let hv = f32_to_halfvec(embedding);
    sqlx::query(
        "INSERT INTO chunks (id, segment_id, chunk_index, leaf_id, segment_index, chunk_label, content_text, embedding, embed_provider, embed_model, embed_dimension)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
         ON CONFLICT (segment_id, chunk_index) DO NOTHING",
    )
    .bind(id)
    .bind(segment_id)
    .bind(chunk_index)
    .bind(leaf_id)
    .bind(segment_index)
    .bind(chunk_label)
    .bind(content_text)
    .bind(hv)
    .bind(embed_provider)
    .bind(embed_model)
    .bind(embed_dimension)
    .execute(pool)
    .await?;
    Ok(())
}

// ── Search ──

#[derive(Debug, serde::Serialize)]
pub struct SearchResult {
    pub chunk_id: Uuid,
    pub leaf_id: String,
    pub segment_index: i32,
    pub chunk_index: i32,
    pub chunk_label: String,
    pub content_text: String,
    pub source_path: String,
    pub score: f64,
}

#[derive(sqlx::FromRow)]
struct SearchRow {
    id: Uuid,
    leaf_id: String,
    segment_index: i32,
    chunk_index: i32,
    chunk_label: String,
    content_text: String,
    source_path: String,
    score: f64,
}

pub async fn search(
    pool: &PgPool,
    query_embedding: &[f32],
    limit: i64,
) -> Result<Vec<SearchResult>> {
    let hv = f32_to_halfvec(query_embedding);
    let rows = sqlx::query_as::<_, SearchRow>(
        "SELECT c.id, c.leaf_id, c.segment_index, c.chunk_index, c.chunk_label,
                c.content_text, l.source_path,
                1.0 - (c.embedding <=> $1::halfvec) AS score
         FROM chunks c
         JOIN leaves l ON l.content_hash = c.leaf_id
         WHERE l.status = 'ready'
         ORDER BY c.embedding <=> $1::halfvec
         LIMIT $2",
    )
    .bind(&hv)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| SearchResult {
            chunk_id: r.id,
            leaf_id: r.leaf_id,
            segment_index: r.segment_index,
            chunk_index: r.chunk_index,
            chunk_label: r.chunk_label,
            content_text: r.content_text,
            source_path: r.source_path,
            score: r.score,
        })
        .collect())
}

// ── Read operations ──

#[derive(Debug, serde::Serialize)]
pub struct ChunkRecord {
    pub id: Uuid,
    pub segment_id: Uuid,
    pub chunk_index: i32,
    pub leaf_id: String,
    pub segment_index: i32,
    pub chunk_label: String,
    pub content_text: String,
}

#[derive(sqlx::FromRow)]
struct ChunkRow {
    id: Uuid,
    segment_id: Uuid,
    chunk_index: i32,
    leaf_id: String,
    segment_index: i32,
    chunk_label: String,
    content_text: String,
}

impl From<ChunkRow> for ChunkRecord {
    fn from(r: ChunkRow) -> Self {
        Self {
            id: r.id,
            segment_id: r.segment_id,
            chunk_index: r.chunk_index,
            leaf_id: r.leaf_id,
            segment_index: r.segment_index,
            chunk_label: r.chunk_label,
            content_text: r.content_text,
        }
    }
}

pub async fn read_chunk(
    pool: &PgPool,
    leaf_id: &str,
    segment_index: i32,
    chunk_index: i32,
) -> Result<Option<ChunkRecord>> {
    let row = sqlx::query_as::<_, ChunkRow>(
        "SELECT id, segment_id, chunk_index, leaf_id, segment_index, chunk_label, content_text
         FROM chunks
         WHERE leaf_id = $1 AND segment_index = $2 AND chunk_index = $3",
    )
    .bind(leaf_id)
    .bind(segment_index)
    .bind(chunk_index)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(Into::into))
}

pub async fn read_chunk_range(
    pool: &PgPool,
    leaf_id: &str,
    segment_index: i32,
    from_chunk: i32,
    to_chunk: i32,
) -> Result<Vec<ChunkRecord>> {
    let rows = sqlx::query_as::<_, ChunkRow>(
        "SELECT id, segment_id, chunk_index, leaf_id, segment_index, chunk_label, content_text
         FROM chunks
         WHERE leaf_id = $1 AND segment_index = $2 AND chunk_index >= $3 AND chunk_index <= $4
         ORDER BY chunk_index",
    )
    .bind(leaf_id)
    .bind(segment_index)
    .bind(from_chunk)
    .bind(to_chunk)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(Into::into).collect())
}

#[derive(Debug, serde::Serialize)]
pub struct SegmentRecord {
    pub id: Uuid,
    pub leaf_id: String,
    pub segment_index: i32,
    pub segment_label: String,
    pub content_text: String,
}

#[derive(sqlx::FromRow)]
struct SegmentRow {
    id: Uuid,
    leaf_id: String,
    segment_index: i32,
    segment_label: String,
    content_text: String,
}

impl From<SegmentRow> for SegmentRecord {
    fn from(r: SegmentRow) -> Self {
        Self {
            id: r.id,
            leaf_id: r.leaf_id,
            segment_index: r.segment_index,
            segment_label: r.segment_label,
            content_text: r.content_text,
        }
    }
}

pub async fn read_segment(
    pool: &PgPool,
    leaf_id: &str,
    segment_index: i32,
) -> Result<Option<SegmentRecord>> {
    let row = sqlx::query_as::<_, SegmentRow>(
        "SELECT id, leaf_id, segment_index, segment_label, content_text
         FROM segments
         WHERE leaf_id = $1 AND segment_index = $2",
    )
    .bind(leaf_id)
    .bind(segment_index)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(Into::into))
}
