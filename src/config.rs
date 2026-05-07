use std::path::PathBuf;

const DEFAULT_DATABASE_URL: &str = "postgresql://josh:ogham@localhost/kosha";

#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub log_level: String,
    pub db_max_connections: u32,
    pub db_acquire_timeout_secs: u64,
    pub db_idle_timeout_secs: u64,
}

impl Config {
    pub fn from_env() -> Self {
        let database_url =
            std::env::var("DATABASE_URL").unwrap_or_else(|_| DEFAULT_DATABASE_URL.to_string());

        let log_level =
            std::env::var("KOSHA_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());

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

        Self {
            database_url,
            log_level,
            db_max_connections,
            db_acquire_timeout_secs,
            db_idle_timeout_secs,
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
