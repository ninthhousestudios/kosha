use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use super::EmbedProvider;
use crate::config::kosha_home;

/// Text-only embedding provider backed by fastembed's ONNX `TextEmbedding`.
///
/// Targets small, CPU-friendly models (MiniLM, bge, nomic, e5, …) for
/// resource-constrained hosts. The concrete model is selected by name via
/// `KOSHA_EMBED_MODEL` (the `EmbeddingModel` variant name, e.g. `AllMiniLML6V2`),
/// and the embedding dimension is derived from the model rather than configured.
pub struct OnnxEmbedder {
    // `TextEmbedding::embed` takes `&mut self` and the provider is shared across
    // async tasks behind `Arc<dyn EmbedProvider>`, so interior mutability is
    // genuinely required here (shared runtime state, not a generics workaround).
    model: Arc<Mutex<TextEmbedding>>,
    model_key: String,
    dim: usize,
    query_prefix: &'static str,
    doc_prefix: &'static str,
    batch_size: usize,
}

impl OnnxEmbedder {
    pub fn load(model_key: &str, batch_size: usize) -> anyhow::Result<Self> {
        let model: EmbeddingModel = model_key.parse().map_err(|e: String| {
            anyhow::anyhow!(
                "{e}. Set KOSHA_EMBED_MODEL to a supported model name, e.g. \
                 AllMiniLML6V2, NomicEmbedTextV15, BGEM3, or MultilingualE5Small"
            )
        })?;

        let dim = TextEmbedding::get_model_info(&model)
            .map_err(|e| anyhow::anyhow!("model info lookup failed: {e}"))?
            .dim;
        let (query_prefix, doc_prefix) = prefixes_for(&model);

        let embedding = TextEmbedding::try_new(
            TextInitOptions::new(model)
                .with_cache_dir(kosha_home().join("models"))
                .with_show_download_progress(true),
        )
        .map_err(|e| anyhow::anyhow!("failed to load ONNX embedding model '{model_key}': {e}"))?;

        Ok(Self {
            model: Arc::new(Mutex::new(embedding)),
            model_key: model_key.to_string(),
            dim,
            query_prefix,
            doc_prefix,
            batch_size,
        })
    }

    /// Embed `texts`, prepending `prefix` to each (the model's retrieval instruction).
    fn embed_prefixed(
        &self,
        texts: Vec<String>,
        prefix: &'static str,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + '_>> {
        let model = Arc::clone(&self.model);
        let batch_size = self.batch_size;
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            tokio::task::spawn_blocking(move || {
                let inputs: Vec<String> = if prefix.is_empty() {
                    texts
                } else {
                    texts.into_iter().map(|t| format!("{prefix}{t}")).collect()
                };
                let mut guard = model
                    .lock()
                    .map_err(|e| anyhow::anyhow!("embedder mutex poisoned: {e}"))?;
                guard
                    .embed(inputs, Some(batch_size))
                    .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))
            })
            .await
            .map_err(|e| anyhow::anyhow!("join error: {e}"))?
        })
    }
}

/// Retrieval prefixes `(query, document)` for models that require them.
///
/// Asymmetric-prefix models (nomic, e5) and query-instruction models (bge-en/zh,
/// mxbai) need these for correct retrieval; everything else (MiniLM, mpnet, gte,
/// bge-m3, …) embeds raw text. mxbai-embed-large-v1 shares bge-en's query prompt
/// (a document prefix would hurt it), so it groups with the bge-en/zh arm.
fn prefixes_for(model: &EmbeddingModel) -> (&'static str, &'static str) {
    use EmbeddingModel::{
        BGEBaseENV15, BGEBaseENV15Q, BGELargeENV15, BGELargeENV15Q, BGELargeZHV15, BGESmallENV15,
        BGESmallENV15Q, BGESmallZHV15, MultilingualE5Base, MultilingualE5Large,
        MultilingualE5Small, MxbaiEmbedLargeV1, MxbaiEmbedLargeV1Q, NomicEmbedTextV1,
        NomicEmbedTextV15, NomicEmbedTextV15Q,
    };
    match model {
        NomicEmbedTextV1 | NomicEmbedTextV15 | NomicEmbedTextV15Q => {
            ("search_query: ", "search_document: ")
        }
        MultilingualE5Small | MultilingualE5Base | MultilingualE5Large => ("query: ", "passage: "),
        BGESmallENV15 | BGESmallENV15Q | BGEBaseENV15 | BGEBaseENV15Q | BGELargeENV15
        | BGELargeENV15Q | BGESmallZHV15 | BGELargeZHV15 | MxbaiEmbedLargeV1
        | MxbaiEmbedLargeV1Q => (
            "Represent this sentence for searching relevant passages: ",
            "",
        ),
        _ => ("", ""),
    }
}

impl EmbedProvider for OnnxEmbedder {
    fn embed_batch(
        &self,
        texts: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + '_>> {
        self.embed_prefixed(texts, self.doc_prefix)
    }

    fn embed_query(
        &self,
        text: String,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<f32>>> + Send + '_>> {
        let fut = self.embed_prefixed(vec![text], self.query_prefix);
        Box::pin(async move {
            fut.await?
                .pop()
                .ok_or_else(|| anyhow::anyhow!("empty embedding result"))
        })
    }

    fn model_name(&self) -> &str {
        &self.model_key
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn provider_name(&self) -> &str {
        "onnx"
    }
}
