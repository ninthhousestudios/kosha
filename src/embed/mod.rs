mod http;
mod local;

pub use candle_core::Device;
pub use http::HttpEmbedder;
pub use local::LocalEmbedder;

use std::future::Future;
use std::pin::Pin;

pub trait EmbedProvider: Send + Sync {
    fn embed_batch(
        &self,
        texts: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + '_>>;

    fn embed_one(
        &self,
        text: String,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<f32>>> + Send + '_>> {
        Box::pin(async move {
            let mut batch = self.embed_batch(vec![text]).await?;
            batch
                .pop()
                .ok_or_else(|| anyhow::anyhow!("empty embedding result"))
        })
    }

    fn embed_image_bytes(
        &self,
        images: Vec<Vec<u8>>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + '_>> {
        let _ = images;
        Box::pin(async { Err(anyhow::anyhow!("image embedding not supported by this provider")) })
    }

    fn model_name(&self) -> &str;
    fn dimension(&self) -> usize;
    fn provider_name(&self) -> &str;
}
