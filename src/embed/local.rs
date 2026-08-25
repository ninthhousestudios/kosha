use std::sync::Arc;

use candle_core::{DType, Device};
use fastembed::Qwen3VLEmbedding;

use super::{BoxFuture, EmbedProvider};

pub struct LocalEmbedder {
    model: Arc<Qwen3VLEmbedding>,
    repo_id: String,
    dim: usize,
}

impl LocalEmbedder {
    pub fn load(repo_id: &str, dimension: usize, device: &Device) -> anyhow::Result<Self> {
        let model = Qwen3VLEmbedding::from_hf(repo_id, device, DType::BF16, 8192)
            .map_err(|e| anyhow::anyhow!("failed to load embedding model: {e}"))?;
        Ok(Self {
            model: Arc::new(model),
            repo_id: repo_id.to_string(),
            dim: dimension,
        })
    }

    /// Embed already-owned texts on the blocking pool. Borrowed callers convert
    /// to `Vec<String>` first because `spawn_blocking` demands a `'static`
    /// closure — the owned copy is a thread-boundary requirement, not a
    /// borrow-checker workaround.
    fn embed_owned(&self, texts: Vec<String>) -> BoxFuture<'_, anyhow::Result<Vec<Vec<f32>>>> {
        let model = Arc::clone(&self.model);
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            tokio::task::spawn_blocking(move || {
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                model
                    .embed_texts(&refs)
                    .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))
            })
            .await
            .map_err(|e| anyhow::anyhow!("join error: {e}"))?
        })
    }
}

impl EmbedProvider for LocalEmbedder {
    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> BoxFuture<'a, anyhow::Result<Vec<Vec<f32>>>> {
        self.embed_owned(texts.iter().map(|&s| s.to_string()).collect())
    }

    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, anyhow::Result<Vec<f32>>> {
        let prefixed = format!(
            "Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery:{text}"
        );
        let fut = self.embed_owned(vec![prefixed]);
        Box::pin(async move {
            fut.await?
                .pop()
                .ok_or_else(|| anyhow::anyhow!("empty embedding result"))
        })
    }

    fn embed_image_bytes(
        &self,
        images: Vec<Vec<u8>>,
    ) -> BoxFuture<'_, anyhow::Result<Vec<Vec<f32>>>> {
        let model = Arc::clone(&self.model);
        Box::pin(async move {
            if images.is_empty() {
                return Ok(vec![]);
            }
            tokio::task::spawn_blocking(move || {
                let refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
                model
                    .embed_image_bytes(&refs)
                    .map_err(|e| anyhow::anyhow!("image embedding failed: {e}"))
            })
            .await
            .map_err(|e| anyhow::anyhow!("join error: {e}"))?
        })
    }

    fn model_name(&self) -> &str {
        &self.repo_id
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn provider_name(&self) -> &str {
        "local"
    }
}
