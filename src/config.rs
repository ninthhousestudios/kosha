use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub log_level: String,
    pub db_max_connections: u32,
    pub db_acquire_timeout_secs: u64,
    pub db_idle_timeout_secs: u64,
    // Embedding
    pub embed_provider: String,
    pub model_repo: String,
    pub embed_url: Option<String>,
    pub embed_model: Option<String>,
    pub embed_dimension: usize,
    pub embed_api_key: Option<String>,
    pub embed_batch_size: usize,
    // Chunking
    pub chunk_target_tokens: usize,
    pub chunk_tolerance_tokens: usize,
    pub chunk_overlap_tokens: usize,
}

impl Config {
    pub fn from_env() -> Self {
        let database_url = std::env::var("DATABASE_URL")
            .expect("DATABASE_URL must be set (via environment or .env file)");

        let log_level = std::env::var("KOSHA_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());

        let db_max_connections: u32 = std::env::var("KOSHA_DB_MAX_CONNECTIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);

        let db_acquire_timeout_secs: u64 = std::env::var("KOSHA_DB_ACQUIRE_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);

        let db_idle_timeout_secs: u64 = std::env::var("KOSHA_DB_IDLE_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

        let embed_provider = std::env::var("KOSHA_EMBED_PROVIDER")
            .unwrap_or_else(|_| "local".to_string());

        let model_repo = std::env::var("KOSHA_MODEL_REPO")
            .unwrap_or_else(|_| "Qwen/Qwen3-VL-Embedding-2B".to_string());

        let embed_url = std::env::var("KOSHA_EMBED_URL").ok();
        let embed_model = std::env::var("KOSHA_EMBED_MODEL").ok();

        let embed_dimension: usize = std::env::var("KOSHA_EMBED_DIMENSION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);

        let embed_api_key = std::env::var("KOSHA_EMBED_API_KEY").ok();

        let embed_batch_size: usize = std::env::var("KOSHA_EMBED_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);

        let chunk_target_tokens: usize = std::env::var("KOSHA_CHUNK_TARGET_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);

        let chunk_tolerance_tokens: usize = std::env::var("KOSHA_CHUNK_TOLERANCE_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);

        let chunk_overlap_tokens: usize = std::env::var("KOSHA_CHUNK_OVERLAP_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        assert!(chunk_target_tokens > 0, "KOSHA_CHUNK_TARGET_TOKENS must be > 0");
        assert!(
            chunk_tolerance_tokens < chunk_target_tokens,
            "KOSHA_CHUNK_TOLERANCE_TOKENS ({chunk_tolerance_tokens}) must be < KOSHA_CHUNK_TARGET_TOKENS ({chunk_target_tokens})"
        );

        Self {
            database_url,
            log_level,
            db_max_connections,
            db_acquire_timeout_secs,
            db_idle_timeout_secs,
            embed_provider,
            model_repo,
            embed_url,
            embed_model,
            embed_dimension,
            embed_api_key,
            embed_batch_size,
            chunk_target_tokens,
            chunk_tolerance_tokens,
            chunk_overlap_tokens,
        }
    }
}

pub fn kosha_home() -> PathBuf {
    if let Some(v) = std::env::var_os("KOSHA_HOME") {
        return PathBuf::from(v);
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".kosha");
    }
    PathBuf::from(".kosha")
}
