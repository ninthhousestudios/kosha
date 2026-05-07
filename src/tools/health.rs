use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::error::Result;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct HealthArgs {}

#[derive(Debug, Serialize)]
pub struct HealthOutput {
    pub status: &'static str,
    pub db_connected: bool,
    pub version: &'static str,
}

#[tracing::instrument(name = "tool.kosha_health", skip(pool))]
pub async fn handle(pool: &PgPool) -> Result<HealthOutput> {
    let db_connected = sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(pool)
        .await
        .is_ok();

    Ok(HealthOutput {
        status: if db_connected { "ok" } else { "degraded" },
        db_connected,
        version: env!("CARGO_PKG_VERSION"),
    })
}
