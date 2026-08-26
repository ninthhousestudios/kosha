# syntax=docker/dockerfile:1

# ── Build stage ────────────────────────────────────────────────────────────
# kosha's `onnx` feature is mutually exclusive with the default `candle-backend`
# (see Cargo.toml), so build with --no-default-features --features onnx. This is
# the CPU-friendly path that matches how the adityas corpus was embedded
# (NomicEmbedTextV15, 768-dim). poppler + cairo are linked by the decompose
# engine even though the /search path never decodes documents.
#
# trixie (glibc 2.41), NOT bookworm (glibc 2.36): the prebuilt ONNX Runtime the
# onnx feature statically links references glibc 2.38+ symbols (__isoc23_strtoll
# et al.), so linking on bookworm fails with "undefined symbol".
FROM rust:1-trixie AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libpoppler-glib-dev libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .
# ONNX Runtime is statically linked under the onnx feature (verified: the binary
# has no libonnxruntime.so dependency), so the runtime stage needs no ORT lib.
RUN cargo build --release --no-default-features --features onnx

# ── Runtime stage ──────────────────────────────────────────────────────────
# The binary dynamically links poppler-glib + cairo (the decompose engine);
# install their runtime packages, which pull glib/gobject/gio transitively.
# Must match the builder's glibc (2.41), so trixie here too.
FROM debian:trixie-slim AS runtime
# curl is used only by the container HEALTHCHECK (the deploy compose probes
# GET /health); libpoppler-glib8 + libcairo2 are the decompose engine's runtime
# deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libpoppler-glib8 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/target/release/kosha /usr/local/bin/kosha

# fastembed downloads the embedding model on first use into KOSHA_HOME/models;
# mount a volume at /data/kosha to persist it across restarts (see ai/36d).
ENV KOSHA_HOME=/data/kosha \
    KOSHA_EMBED_PROVIDER=onnx \
    KOSHA_EMBED_MODEL=NomicEmbedTextV15 \
    KOSHA_HTTP_ADDR=0.0.0.0 \
    KOSHA_HTTP_PORT=3400 \
    RUST_LOG=info
RUN mkdir -p /data/kosha
VOLUME ["/data/kosha"]

EXPOSE 3400
# DATABASE_URL is injected by compose (points at the pgvector Postgres holding
# the restored corpus). --device cpu: no CUDA in-container; onnx is CPU-only.
ENTRYPOINT ["kosha", "serve", "--http", "--device", "cpu"]
