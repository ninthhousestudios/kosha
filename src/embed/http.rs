use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use super::EmbedProvider;

pub struct HttpEmbedder {
    client: reqwest::Client,
    url: String,
    model: String,
    dim: usize,
    api_key: Option<String>,
    batch_size: usize,
}

impl HttpEmbedder {
    pub fn new(
        url: String,
        model: String,
        dimension: usize,
        api_key: Option<String>,
        batch_size: usize,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");
        Self {
            client,
            url,
            model,
            dim: dimension,
            api_key,
            batch_size,
        }
    }
}

#[derive(serde::Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
    #[serde(default)]
    error: Option<EmbedErrorBody>,
}

#[derive(serde::Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(serde::Deserialize)]
struct EmbedErrorBody {
    message: String,
}

impl EmbedProvider for HttpEmbedder {
    fn embed_batch(
        &self,
        texts: Vec<String>,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<Vec<f32>>>> + Send + '_>> {
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(vec![]);
            }

            let mut all = Vec::with_capacity(texts.len());

            for batch_start in (0..texts.len()).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(texts.len());
                let batch: Vec<String> = texts[batch_start..batch_end].to_vec();
                let batch_len = batch.len();

                let mut req = self.client.post(&self.url).json(&EmbedRequest {
                    model: self.model.clone(),
                    input: batch,
                });

                if let Some(ref key) = self.api_key {
                    req = req.bearer_auth(key);
                }

                let resp = req
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("embedding HTTP request failed: {e}"))?;

                let status = resp.status();
                if !status.is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!(
                        "embedding API returned {status}: {body}"
                    ));
                }

                let body: EmbedResponse = resp
                    .json()
                    .await
                    .map_err(|e| anyhow::anyhow!("failed to parse embedding response: {e}"))?;

                if let Some(err) = body.error {
                    return Err(anyhow::anyhow!("embedding API error: {}", err.message));
                }

                let mut sorted = vec![vec![]; batch_len];
                for d in body.data {
                    if d.index < sorted.len() {
                        sorted[d.index] = d.embedding;
                    }
                }

                for (i, emb) in sorted.iter().enumerate() {
                    if emb.is_empty() {
                        return Err(anyhow::anyhow!(
                            "missing embedding for index {}",
                            batch_start + i
                        ));
                    }
                }

                all.extend(sorted);
            }

            Ok(all)
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn provider_name(&self) -> &str {
        "http"
    }
}
