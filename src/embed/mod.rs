mod http;
#[cfg(feature = "candle-backend")]
mod local;
#[cfg(feature = "onnx")]
mod onnx;

#[cfg(feature = "candle-backend")]
pub use candle_core::Device;
pub use http::HttpEmbedder;
#[cfg(feature = "candle-backend")]
pub use local::LocalEmbedder;
#[cfg(feature = "onnx")]
pub use onnx::OnnxEmbedder;

use std::future::Future;
use std::pin::Pin;

pub trait EmbedProvider: Send + Sync {
    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + 'a>>;

    fn embed_one<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<f32>>> + Send + 'a>> {
        Box::pin(async move {
            let mut batch = self.embed_batch(&[text]).await?;
            batch
                .pop()
                .ok_or_else(|| anyhow::anyhow!("empty embedding result"))
        })
    }

    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<f32>>> + Send + 'a>> {
        self.embed_one(text)
    }

    fn embed_image_bytes(
        &self,
        images: Vec<Vec<u8>>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + '_>> {
        let _ = images;
        Box::pin(async {
            Err(anyhow::anyhow!(
                "image embedding not supported by this provider"
            ))
        })
    }

    fn model_name(&self) -> &str;
    fn dimension(&self) -> usize;
    fn provider_name(&self) -> &str;
}
