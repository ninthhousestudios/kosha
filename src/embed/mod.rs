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

/// A `Send` boxed future, borrowing for `'a`. Every `EmbedProvider` method
/// returns one: the trait predates native `async fn` in traits, so the futures
/// are hand-boxed.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait EmbedProvider: Send + Sync {
    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> BoxFuture<'a, anyhow::Result<Vec<Vec<f32>>>>;

    fn embed_one<'a>(&'a self, text: &'a str) -> BoxFuture<'a, anyhow::Result<Vec<f32>>> {
        Box::pin(async move {
            let mut batch = self.embed_batch(&[text]).await?;
            batch
                .pop()
                .ok_or_else(|| anyhow::anyhow!("empty embedding result"))
        })
    }

    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, anyhow::Result<Vec<f32>>> {
        self.embed_one(text)
    }

    fn embed_image_bytes(
        &self,
        images: Vec<Vec<u8>>,
    ) -> BoxFuture<'_, anyhow::Result<Vec<Vec<f32>>>> {
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
