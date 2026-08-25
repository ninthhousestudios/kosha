// `onnx` and `candle-backend` select conflicting `ort` backends: `onnx` links a
// downloaded ONNX Runtime (`ort/download-binaries`), while `candle-backend`
// disables ort linking (`ort/alternative-backend`). Enabling both produces an
// opaque ort feature/link error; surface the real cause here instead.
#[cfg(all(feature = "onnx", feature = "candle-backend"))]
compile_error!(
    "features `onnx` and `candle-backend` are mutually exclusive (conflicting `ort` backends); \
     build the ONNX provider with `--no-default-features --features onnx`"
);

pub mod chunk;
pub mod config;
pub mod db;
pub mod decompose;
pub mod embed;
pub mod error;
pub mod ingest;
pub mod mcp;
pub mod serve_http;
pub mod store;
pub mod tools;
