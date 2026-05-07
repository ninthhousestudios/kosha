use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use candle_core::{DType, Device};
use fastembed::Qwen3VLEmbedding;

use super::EmbedProvider;

pub struct LocalEmbedder {
    model: Arc<Qwen3VLEmbedding>,
    repo_id: String,
    dim: usize,
}

impl LocalEmbedder {
    pub fn load(repo_id: &str, dimension: usize) -> anyhow::Result<Self> {
        let model = Qwen3VLEmbedding::from_hf(repo_id, &Device::Cpu, DType::BF16, 8192)
            .map_err(|e| anyhow::anyhow!("failed to load embedding model: {e}"))?;
        Ok(Self {
            model: Arc::new(model),
            repo_id: repo_id.to_string(),
            dim: dimension,
        })
    }
}

impl EmbedProvider for LocalEmbedder {
    fn embed_batch(
        &self,
        texts: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + '_>> {
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

    fn embed_image_bytes(
        &self,
        images: Vec<Vec<u8>>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + '_>> {
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
