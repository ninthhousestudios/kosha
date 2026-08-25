use std::sync::{Arc, Mutex};

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

use super::{BoxFuture, EmbedProvider};
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
    ///
    /// `inputs` is built up front (synchronously) so the returned future owns its
    /// data: `spawn_blocking` demands a `'static` closure, and reading the borrowed
    /// `texts` before crossing that boundary is a thread requirement, not a
    /// borrow-checker workaround.
    fn embed_prefixed(
        &self,
        texts: &[&str],
        prefix: &'static str,
    ) -> BoxFuture<'_, anyhow::Result<Vec<Vec<f32>>>> {
        let model = Arc::clone(&self.model);
        let batch_size = self.batch_size;
        let inputs: Vec<String> = if prefix.is_empty() {
            texts.iter().map(|&t| t.to_string()).collect()
        } else {
            texts.iter().map(|&t| format!("{prefix}{t}")).collect()
        };
        Box::pin(async move {
            if inputs.is_empty() {
                return Ok(vec![]);
            }
            tokio::task::spawn_blocking(move || {
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
/// Three shapes appear here, each verified against the model card:
/// - **Asymmetric** — distinct query/document prefixes. nomic and modernbert-embed
///   (which follows the nomic recipe) use `search_query:`/`search_document:`; e5 uses
///   `query:`/`passage:`; embeddinggemma uses its own task-tagged prompts.
/// - **Query-instruction only** — a query prompt, no document prefix (a document
///   prefix would hurt these). bge-en/zh, mxbai-embed-large-v1, and the whole
///   snowflake-arctic-embed family share the "Represent this sentence…" prompt.
/// - **Raw text** — no prefix (the default arm): MiniLM, mpnet, gte-en-v1.5, bge-m3,
///   jina-embeddings-v2, clip, and any model not listed above.
fn prefixes_for(model: &EmbeddingModel) -> (&'static str, &'static str) {
    use EmbeddingModel::{
        BGEBaseENV15, BGEBaseENV15Q, BGELargeENV15, BGELargeENV15Q, BGELargeZHV15, BGESmallENV15,
        BGESmallENV15Q, BGESmallZHV15, EmbeddingGemma300M, ModernBertEmbedLarge,
        MultilingualE5Base, MultilingualE5Large, MultilingualE5Small, MxbaiEmbedLargeV1,
        MxbaiEmbedLargeV1Q, NomicEmbedTextV1, NomicEmbedTextV15, NomicEmbedTextV15Q,
        SnowflakeArcticEmbedL, SnowflakeArcticEmbedLQ, SnowflakeArcticEmbedM,
        SnowflakeArcticEmbedMLong, SnowflakeArcticEmbedMLongQ, SnowflakeArcticEmbedMQ,
        SnowflakeArcticEmbedS, SnowflakeArcticEmbedSQ, SnowflakeArcticEmbedXS,
        SnowflakeArcticEmbedXSQ,
    };
    match model {
        NomicEmbedTextV1 | NomicEmbedTextV15 | NomicEmbedTextV15Q | ModernBertEmbedLarge => {
            ("search_query: ", "search_document: ")
        }
        MultilingualE5Small | MultilingualE5Base | MultilingualE5Large => ("query: ", "passage: "),
        EmbeddingGemma300M => ("task: search result | query: ", "title: none | text: "),
        BGESmallENV15
        | BGESmallENV15Q
        | BGEBaseENV15
        | BGEBaseENV15Q
        | BGELargeENV15
        | BGELargeENV15Q
        | BGESmallZHV15
        | BGELargeZHV15
        | MxbaiEmbedLargeV1
        | MxbaiEmbedLargeV1Q
        | SnowflakeArcticEmbedXS
        | SnowflakeArcticEmbedXSQ
        | SnowflakeArcticEmbedS
        | SnowflakeArcticEmbedSQ
        | SnowflakeArcticEmbedM
        | SnowflakeArcticEmbedMQ
        | SnowflakeArcticEmbedMLong
        | SnowflakeArcticEmbedMLongQ
        | SnowflakeArcticEmbedL
        | SnowflakeArcticEmbedLQ => (
            "Represent this sentence for searching relevant passages: ",
            "",
        ),
        _ => ("", ""),
    }
}

impl EmbedProvider for OnnxEmbedder {
    fn embed_batch<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> BoxFuture<'a, anyhow::Result<Vec<Vec<f32>>>> {
        self.embed_prefixed(texts, self.doc_prefix)
    }

    fn embed_query<'a>(&'a self, text: &'a str) -> BoxFuture<'a, anyhow::Result<Vec<f32>>> {
        let fut = self.embed_prefixed(&[text], self.query_prefix);
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

#[cfg(test)]
mod tests {
    use super::prefixes_for;
    use fastembed::EmbeddingModel;

    #[test]
    fn nomic_and_modernbert_share_search_prefixes() {
        // modernbert-embed follows the nomic recipe, so it shares the arm.
        for m in [
            EmbeddingModel::NomicEmbedTextV15,
            EmbeddingModel::ModernBertEmbedLarge,
        ] {
            assert_eq!(
                prefixes_for(&m),
                ("search_query: ", "search_document: "),
                "{m:?}"
            );
        }
    }

    #[test]
    fn e5_uses_query_passage() {
        assert_eq!(
            prefixes_for(&EmbeddingModel::MultilingualE5Small),
            ("query: ", "passage: ")
        );
    }

    #[test]
    fn embeddinggemma_uses_task_tagged_prompts() {
        assert_eq!(
            prefixes_for(&EmbeddingModel::EmbeddingGemma300M),
            ("task: search result | query: ", "title: none | text: ")
        );
    }

    #[test]
    fn bge_mxbai_and_snowflake_are_query_instruction_only() {
        let expected = (
            "Represent this sentence for searching relevant passages: ",
            "",
        );
        for m in [
            EmbeddingModel::BGESmallENV15,
            EmbeddingModel::MxbaiEmbedLargeV1,
            EmbeddingModel::SnowflakeArcticEmbedXS,
            EmbeddingModel::SnowflakeArcticEmbedMLong,
            EmbeddingModel::SnowflakeArcticEmbedLQ,
        ] {
            assert_eq!(prefixes_for(&m), expected, "{m:?}");
        }
    }

    #[test]
    fn raw_text_models_get_no_prefix() {
        // gte-en-v1.5, jina-v2, and clip embed raw text — verified against their cards.
        for m in [
            EmbeddingModel::AllMiniLML6V2,
            EmbeddingModel::AllMpnetBaseV2,
            EmbeddingModel::BGEM3,
            EmbeddingModel::GTELargeENV15,
            EmbeddingModel::JinaEmbeddingsV2BaseEN,
            EmbeddingModel::ClipVitB32,
        ] {
            assert_eq!(prefixes_for(&m), ("", ""), "{m:?}");
        }
    }
}
