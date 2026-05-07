use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, Serialize)]
pub struct ErrorData {
    pub tool: &'static str,
    pub constraint: String,
    pub next_action: String,
}

#[derive(Debug, Error)]
pub enum KoshaError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),

    #[error("migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),

    #[error("not found: {kind}")]
    NotFound {
        tool: &'static str,
        kind: &'static str,
        next_action: String,
    },

    #[error("internal error: {message}")]
    Internal { tool: &'static str, message: String },

    #[error("embed error: {0}")]
    Embed(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl KoshaError {
    pub fn code(&self) -> i32 {
        match self {
            Self::NotFound { .. } => -32602,
            _ => -32603,
        }
    }

    pub fn data(&self) -> ErrorData {
        match self {
            Self::Db(e) => ErrorData {
                tool: "server",
                constraint: "Postgres must be reachable".to_string(),
                next_action: format!("Check DATABASE_URL and Postgres status: {e}"),
            },
            Self::Migration(e) => ErrorData {
                tool: "server",
                constraint: "migrations must succeed".to_string(),
                next_action: format!("Check migration files: {e}"),
            },
            Self::NotFound {
                tool,
                kind,
                next_action,
            } => ErrorData {
                tool,
                constraint: format!("{kind} must exist"),
                next_action: next_action.clone(),
            },
            Self::Internal { tool, message } => ErrorData {
                tool,
                constraint: message.clone(),
                next_action: "Report this as a bug; include server logs.".to_string(),
            },
            Self::Embed(msg) => ErrorData {
                tool: "server",
                constraint: "embedding model must be operational".to_string(),
                next_action: format!("Check model availability: {msg}"),
            },
            Self::Io(e) => ErrorData {
                tool: "server",
                constraint: "file must be readable".to_string(),
                next_action: format!("Check file path and permissions: {e}"),
            },
        }
    }
}

pub type Result<T> = std::result::Result<T, KoshaError>;
