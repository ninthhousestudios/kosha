use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::error::Result;
use crate::store;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CollectionsArgs {}

#[derive(Debug, Serialize)]
pub struct CollectionsOutput {
    pub collections: Vec<String>,
    pub count: usize,
}

#[tracing::instrument(name = "tool.kosha_collections", skip(pool))]
pub async fn handle(pool: &PgPool, _args: CollectionsArgs) -> Result<CollectionsOutput> {
    let collections = store::list_collections(pool).await?;
    let count = collections.len();
    Ok(CollectionsOutput { collections, count })
}
