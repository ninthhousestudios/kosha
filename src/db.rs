use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

use crate::config::Config;
use crate::error::{KoshaError, Result};

pub async fn create_pool(cfg: &Config) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(cfg.db_max_connections)
        .acquire_timeout(std::time::Duration::from_secs(cfg.db_acquire_timeout_secs))
        .idle_timeout(std::time::Duration::from_secs(cfg.db_idle_timeout_secs))
        .connect(&cfg.database_url)
        .await?;

    Ok(pool)
}

pub async fn run_migrations(pool: &PgPool) -> Result<()> {
    sqlx::migrate!("./migrations").run(pool).await?;
    Ok(())
}

/// Reconcile the `chunks.embedding` column dimension with the active embedder.
///
/// The initial migration creates `halfvec(2048)` for the multimodal default. When a
/// different model is configured (e.g. a small ONNX text model), the column and its
/// HNSW index are recreated at the model's dimension. This only rewrites an *unused*
/// embedding column: if chunks already hold embeddings at another dimension it bails
/// rather than silently destroy them. A no-op when the dimension already matches.
pub async fn ensure_embedding_dim(pool: &PgPool, dim: usize) -> Result<()> {
    // halfvec HNSW supports up to 4000 dims; bound the value we interpolate below.
    if !(1..=4000).contains(&dim) {
        return Err(KoshaError::Internal {
            tool: "server",
            message: format!("embedding dimension {dim} out of range for halfvec HNSW (1..=4000)"),
        });
    }

    // e.g. "halfvec(2048)"; None if the table/column is absent.
    let current: Option<String> = sqlx::query_scalar(
        "SELECT format_type(a.atttypid, a.atttypmod)
         FROM pg_attribute a
         JOIN pg_class c ON c.oid = a.attrelid
         WHERE c.relname = 'chunks' AND a.attname = 'embedding' AND NOT a.attisdropped",
    )
    .fetch_optional(pool)
    .await?;

    let Some(current) = current else {
        return Ok(());
    };

    let current_dim: Option<usize> = current
        .split_once('(')
        .and_then(|(_, rest)| rest.strip_suffix(')'))
        .and_then(|n| n.parse().ok());
    if current_dim == Some(dim) {
        return Ok(());
    }

    let existing: i64 =
        sqlx::query_scalar("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            .fetch_one(pool)
            .await?;
    if existing > 0 {
        return Err(KoshaError::Internal {
            tool: "server",
            message: format!(
                "chunks already holds {existing} embedding(s) as {current}, but the active model \
                 needs dimension {dim}. Use a fresh database (different DATABASE_URL) or clear \
                 embeddings before switching models."
            ),
        });
    }

    tracing::info!(from = %current, to = dim, "recreating embedding column at new dimension");
    // `dim` is validated to 1..=4000 above, so interpolation is safe.
    sqlx::query("DROP INDEX IF EXISTS chunks_embedding_idx")
        .execute(pool)
        .await?;
    sqlx::query(&format!(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE halfvec({dim})"
    ))
    .execute(pool)
    .await?;
    sqlx::query(&format!(
        "ALTER TABLE chunks ALTER COLUMN embed_dimension SET DEFAULT {dim}"
    ))
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding halfvec_cosine_ops) \
         WITH (m = 16, ef_construction = 64)",
    )
    .execute(pool)
    .await?;

    Ok(())
}
