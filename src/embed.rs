use std::sync::Arc;

use candle_core::{DType, Device};
use fastembed::Qwen3TextEmbedding;

pub struct Embedder {
    model: Qwen3TextEmbedding,
}

impl Embedder {
    pub fn load(repo_id: &str) -> anyhow::Result<Self> {
        let model = Qwen3TextEmbedding::from_hf(repo_id, &Device::Cpu, DType::BF16, 8192)
            .map_err(|e| anyhow::anyhow!("failed to load embedding model: {e}"))?;
        Ok(Self { model })
    }

    pub fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.model
            .embed(texts)
            .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))
    }

    pub fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut batch = self.embed_batch(&[text])?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("empty embedding result"))
    }

    pub async fn embed_one_async(self: &Arc<Self>, text: String) -> anyhow::Result<Vec<f32>> {
        let this = Arc::clone(self);
        tokio::task::spawn_blocking(move || this.embed_one(&text))
            .await
            .map_err(|e| anyhow::anyhow!("join error: {e}"))?
    }

    pub async fn embed_batch_async(
        self: &Arc<Self>,
        texts: Vec<String>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let this = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            this.embed_batch(&refs)
        })
        .await
        .map_err(|e| anyhow::anyhow!("join error: {e}"))?
    }
}
