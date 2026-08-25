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
    /// Effective output dimension — the model's native dim, or the Matryoshka
    /// truncation target when one is configured. What `dimension()` reports.
    dim: usize,
    /// `Some(n)` to truncate every embedding to its first `n` components and
    /// L2-renormalize (Matryoshka); `None` to emit the native vector unchanged.
    truncate: Option<usize>,
    query_prefix: &'static str,
    doc_prefix: &'static str,
    batch_size: usize,
}

impl OnnxEmbedder {
    /// Load a fastembed model. `dim_override` is the explicitly-set
    /// `KOSHA_EMBED_DIMENSION`; when present and below the model's native
    /// dimension it requests Matryoshka truncation (rejected for models that
    /// are not Matryoshka-trained — see [`resolve_dim`]).
    pub fn load(
        model_key: &str,
        batch_size: usize,
        dim_override: Option<usize>,
    ) -> anyhow::Result<Self> {
        let model: EmbeddingModel = model_key.parse().map_err(|e: String| {
            anyhow::anyhow!(
                "{e}. Set KOSHA_EMBED_MODEL to a supported model name, e.g. \
                 AllMiniLML6V2, NomicEmbedTextV15, BGEM3, or MultilingualE5Small"
            )
        })?;

        let native_dim = TextEmbedding::get_model_info(&model)
            .map_err(|e| anyhow::anyhow!("model info lookup failed: {e}"))?
            .dim;
        let (dim, truncate) = resolve_dim(&model, native_dim, dim_override, model_key)?;
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
            truncate,
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
        let truncate = self.truncate;
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
                let vectors = guard
                    .embed(inputs, Some(batch_size))
                    .map_err(|e| anyhow::anyhow!("embedding failed: {e}"))?;
                Ok(match truncate {
                    Some(dim) => vectors.into_iter().map(|v| truncate_l2(v, dim)).collect(),
                    None => vectors,
                })
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

/// fastembed models trained with Matryoshka Representation Learning, whose
/// output vectors stay meaningful when truncated to a shorter prefix and
/// renormalized. Verified against each model card. Truncating any other model
/// scrambles the vector, so [`resolve_dim`] refuses it.
fn supports_matryoshka(model: &EmbeddingModel) -> bool {
    use EmbeddingModel::{
        EmbeddingGemma300M, ModernBertEmbedLarge, MxbaiEmbedLargeV1, MxbaiEmbedLargeV1Q,
        NomicEmbedTextV15, NomicEmbedTextV15Q,
    };
    // nomic v1 predates MRL — only v1.5 qualifies.
    matches!(
        model,
        NomicEmbedTextV15
            | NomicEmbedTextV15Q
            | MxbaiEmbedLargeV1
            | MxbaiEmbedLargeV1Q
            | ModernBertEmbedLarge
            | EmbeddingGemma300M
    )
}

/// Resolve the effective output dimension and optional truncation target from a
/// requested `dim_override` against the model's `native_dim`.
///
/// Returns `(effective_dim, truncate)`: `truncate` is `Some(n)` only when the
/// vectors must be shortened at runtime. A truncation below native is allowed
/// only for Matryoshka-trained models — for anything else it is a hard error,
/// because the alternative is silently persisting a corpus of garbage vectors.
fn resolve_dim(
    model: &EmbeddingModel,
    native_dim: usize,
    dim_override: Option<usize>,
    model_key: &str,
) -> anyhow::Result<(usize, Option<usize>)> {
    let Some(req) = dim_override else {
        return Ok((native_dim, None));
    };
    if req == native_dim {
        // Explicitly pinned to native — no truncation, and no capability check.
        return Ok((native_dim, None));
    }
    if req == 0 {
        anyhow::bail!("KOSHA_EMBED_DIMENSION must be >= 1");
    }
    if req > native_dim {
        anyhow::bail!(
            "KOSHA_EMBED_DIMENSION={req} exceeds the native dimension {native_dim} of \
             '{model_key}'; the ONNX provider cannot upscale embeddings. Unset it to use \
             the native {native_dim}."
        );
    }
    if !supports_matryoshka(model) {
        anyhow::bail!(
            "KOSHA_EMBED_DIMENSION={req} would truncate '{model_key}' below its native \
             {native_dim}, but that model is not Matryoshka-trained — truncating it yields \
             meaningless vectors. Truncatable models: NomicEmbedTextV15, MxbaiEmbedLargeV1, \
             ModernBertEmbedLarge, EmbeddingGemma300M. Unset KOSHA_EMBED_DIMENSION to use \
             the native {native_dim}."
        );
    }
    Ok((req, Some(req)))
}

/// Truncate a Matryoshka embedding to its first `dim` components and
/// L2-renormalize. `dim` is guaranteed `<= v.len()` by [`resolve_dim`].
fn truncate_l2(mut v: Vec<f32>, dim: usize) -> Vec<f32> {
    v.truncate(dim);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
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
    use super::{prefixes_for, resolve_dim, truncate_l2};
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

    #[test]
    fn resolve_dim_no_override_uses_native() {
        let r = resolve_dim(
            &EmbeddingModel::NomicEmbedTextV15,
            768,
            None,
            "NomicEmbedTextV15",
        );
        assert_eq!(r.unwrap(), (768, None));
    }

    #[test]
    fn resolve_dim_override_equal_to_native_is_noop() {
        // Even a non-Matryoshka model may be pinned to its own native dim.
        let r = resolve_dim(
            &EmbeddingModel::AllMiniLML6V2,
            384,
            Some(384),
            "AllMiniLML6V2",
        );
        assert_eq!(r.unwrap(), (384, None));
    }

    #[test]
    fn resolve_dim_truncates_matryoshka_model() {
        let r = resolve_dim(
            &EmbeddingModel::NomicEmbedTextV15,
            768,
            Some(256),
            "NomicEmbedTextV15",
        );
        assert_eq!(r.unwrap(), (256, Some(256)));
    }

    #[test]
    fn resolve_dim_rejects_truncating_non_matryoshka() {
        // nomic v1 (not v1.5) is deliberately excluded from the whitelist.
        let r = resolve_dim(
            &EmbeddingModel::NomicEmbedTextV1,
            768,
            Some(256),
            "NomicEmbedTextV1",
        );
        assert!(r.is_err());
    }

    #[test]
    fn resolve_dim_rejects_upscale_and_zero() {
        assert!(
            resolve_dim(&EmbeddingModel::NomicEmbedTextV15, 768, Some(1024), "m").is_err(),
            "upscale must be rejected"
        );
        assert!(
            resolve_dim(&EmbeddingModel::NomicEmbedTextV15, 768, Some(0), "m").is_err(),
            "zero must be rejected"
        );
    }

    #[test]
    fn truncate_l2_shortens_and_normalizes() {
        let out = truncate_l2(vec![3.0, 4.0, 100.0, -7.0], 2);
        assert_eq!(out.len(), 2);
        // [3,4] normalized -> [0.6, 0.8]
        assert!((out[0] - 0.6).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 0.8).abs() < 1e-6, "{out:?}");
        let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "unit norm, got {norm}");
    }

    #[test]
    fn truncate_l2_handles_zero_vector() {
        // A degenerate all-zero vector must not divide by zero.
        let out = truncate_l2(vec![0.0, 0.0, 0.0], 2);
        assert_eq!(out, vec![0.0, 0.0]);
    }
}
