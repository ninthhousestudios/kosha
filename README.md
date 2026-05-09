# kosha

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

Document intelligence for the [manas](https://github.com/ninthhousestudios/manas) ecosystem. Decomposes documents into segments, embeds them into a shared vector space, and exposes semantic search and citation tools over MCP.

kosha handles PDFs (including scanned pages with no text layer), EPUBs, Markdown, HTML, plain text, and images. The embedding model ([Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)) encodes both text and images into the same 2048-dim vector space, so a text query can retrieve a scanned page directly.

## Prerequisites

- **Postgres** with [pgvector](https://github.com/pgvector/pgvector) extension
- **System libraries**: libpoppler-glib-dev, libcairo2-dev (for PDF rendering)
- **Rust** 1.85+ (edition 2024)

## Quick start

```bash
# Create the database
createdb kosha
psql -d kosha -c 'CREATE EXTENSION IF NOT EXISTS vector;'

# Set the connection string
export DATABASE_URL="postgresql://user:pass@localhost/kosha"

# Build
cargo build --release

# Ingest some documents
kosha ingest paper.pdf
kosha ingest -r ~/library/ --collection classics --tag reference

# Browse what's ingested
kosha list
kosha list --collection classics
kosha list a3f7b2  # outline for a specific document (hash prefix)

# Run as MCP server
kosha serve
```

## CLI reference

### `kosha ingest`

Ingest files into the database. Each file is decomposed into segments, chunked, and embedded.

```
kosha ingest [OPTIONS] <PATHS>...
```

| Argument / Flag | Description |
|---|---|
| `<PATHS>...` | Files or directories to ingest |
| `-r, --recursive` | Recurse into directories |
| `--collection <NAME>` | Assign to a collection (default: `default`) |
| `--tag <TAG>` | Attach a tag (repeatable) |

Files already ingested (same BLAKE3 hash) are skipped automatically.

**Supported formats**: PDF, EPUB, PNG, JPG, Markdown, HTML, plain text.

### `kosha list`

List ingested documents or show the segment outline for a specific one.

```
kosha list [OPTIONS] [LEAF]
```

| Argument / Flag | Description |
|---|---|
| `[LEAF]` | Content hash or prefix to show outline for |
| `--collection <NAME>` | Filter by collection |
| `--format <FORMAT>` | Filter by format (e.g. `pdf`, `epub`, `markdown`) |
| `--tag <TAG>` | Filter by tag (repeatable) |

Without `[LEAF]`, lists all documents with hash, format, collection, segment/chunk counts, and path. With a hash prefix, shows the segment outline for that document.

### `kosha serve`

Run as a stdio MCP server. This is how agents interact with kosha.

```
kosha serve [OPTIONS]
```

Exposes these MCP tools:

| Tool | Purpose |
|---|---|
| `kosha_search` | Semantic search across all segments |
| `kosha_read` | Fetch a segment by leaf hash and segment index |
| `kosha_outline` | Segment outline for a document |
| `kosha_leaf` | Metadata for a single document |
| `kosha_leaves` | List documents with filters |
| `kosha_collections` | List collections |
| `kosha_health` | Status check |

### Global options

| Flag | Description |
|---|---|
| `--device <DEVICE>` | Device for local embedding: `cpu`, `gpu`, `auto` (default: `auto`) |
| `-h, --help` | Print help |
| `-V, --version` | Print version |

The `--device` flag controls where the embedding model runs. `auto` tries CUDA if available, falls back to CPU. `gpu` requires building with the `cuda` feature (see below).

## Configuration

All configuration is via environment variables. Place them in a `.env` file in the project directory or in `~/.kosha/.env`.

### Required

| Variable | Description |
|---|---|
| `DATABASE_URL` | Postgres connection string |

### Embedding

| Variable | Default | Description |
|---|---|---|
| `KOSHA_EMBED_PROVIDER` | `local` | `local` (on-device) or `http` (remote API) |
| `KOSHA_MODEL_REPO` | `Qwen/Qwen3-VL-Embedding-2B` | HuggingFace repo for local provider |
| `KOSHA_EMBED_DIMENSION` | `2048` | Embedding dimension |
| `KOSHA_EMBED_URL` | — | API endpoint (required for `http` provider) |
| `KOSHA_EMBED_MODEL` | — | Model name (required for `http` provider) |
| `KOSHA_EMBED_API_KEY` | — | Bearer token for `http` provider |
| `KOSHA_EMBED_BATCH_SIZE` | `32` | Batch size for `http` provider |

### Chunking

| Variable | Default | Description |
|---|---|---|
| `KOSHA_CHUNK_TARGET_TOKENS` | `512` | Target chunk size in tokens |
| `KOSHA_CHUNK_TOLERANCE_TOKENS` | `256` | Allowed deviation from target |
| `KOSHA_CHUNK_OVERLAP_TOKENS` | `0` | Overlap between consecutive chunks |

### Other

| Variable | Default | Description |
|---|---|---|
| `KOSHA_LOG_LEVEL` | `info` | Log level filter |
| `KOSHA_HOME` | `~/.kosha` | Config/data directory |
| `KOSHA_DB_MAX_CONNECTIONS` | `8` | Connection pool size |
| `KOSHA_DB_ACQUIRE_TIMEOUT_SECS` | `5` | Pool acquire timeout |
| `KOSHA_DB_IDLE_TIMEOUT_SECS` | `300` | Idle connection timeout |

## GPU support

By default, kosha embeds on CPU using candle with BF16 matrix multiplication. For faster embedding on NVIDIA GPUs, build with the `cuda` feature:

```bash
cargo build --release --features cuda
kosha --device gpu ingest -r /path/to/corpus/
```

This requires the CUDA toolkit. See [docs/cloud-ingest.md](docs/cloud-ingest.md) for a full RunPod workflow: deploy, ingest on GPU, `pg_dump`, restore locally.

## How it works

1. **Decompose** — each file is split into format-native segments. PDFs split by page, EPUBs by TOC entry, Markdown/HTML by heading hierarchy, images as a single segment.

2. **Chunk** — text segments are split into chunks targeting ~512 tokens, respecting document structure (headings, paragraphs, code fences).

3. **Embed** — each chunk is embedded via Qwen3-VL into a 2048-dim vector. Image segments (scanned PDF pages, standalone images) are embedded directly from pixels.

4. **Store** — segments, chunks, and vectors land in Postgres with pgvector. Documents are keyed by BLAKE3 content hash, so re-ingesting the same file is a no-op.

5. **Search** — queries are embedded with the same model and matched against stored vectors via HNSW index. Text and image segments live in the same vector space.

## Project layout

```
src/
  main.rs          CLI entry point
  config.rs        Environment-based configuration
  db.rs            Connection pool and migrations
  store.rs         Database read/write operations
  ingest.rs        File walking and ingestion pipeline
  chunk.rs         Chunking with structure awareness
  mcp.rs           MCP server wiring
  decompose/       Format-specific decomposers
    pdf.rs         PDF pages (text + rendered images)
    epub.rs        EPUB with TOC-derived labels
    html.rs        HTML heading hierarchy
    markdown.rs    Markdown heading hierarchy
    image.rs       PNG/JPG as single segments
    plain_text.rs  Fallback plain text
  embed/           Embedding providers
    local.rs       On-device via fastembed + candle
    http.rs        Remote API (OpenAI-compatible)
  tools/           MCP tool implementations
migrations/        SQL migrations (run automatically)
docs/
  architecture.md  Design document
  cloud-ingest.md  RunPod GPU workflow
```

## License

MIT OR Apache-2.0
