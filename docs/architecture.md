# kosha — architecture

Status: design
Date: 2026-04-29
Supersedes: `kosha-sketch.md` (2026-04-27)
Companion to: `../../smriti/docs/smriti-kosha-architecture-sketch.md` (the interface between smriti and kosha)

## what kosha is

kosha (कोश — treasury) is the document comprehension layer for the manas ecosystem. It decomposes documents into segments, embeds them into a shared vector space, and exposes semantic search and citation tools over MCP.

smriti finds the file. kosha reads what's inside.

## what kosha is not

- Not a filesystem tracker. smriti does that. kosha subscribes to smriti's event stream.
- Not a general-purpose search engine. It searches *within* documents that smriti already knows about.
- Not an OCR pipeline (v1). For scan-only pages, the page image is embedded directly via Qwen3-VL. No text extraction from images.
- Not a code intelligence tool. That's sutra.
- Not a memory system. That's chitta. kosha provides citations that chitta can store.

## core concepts

### book

A **book** is any ingested document — a PDF, epub, article, research paper, plain text file, HTML page. One file = one book record. The term is a convenient label, not a statement about the content's nature.

Identity: each book is keyed by its `content_hash` (BLAKE3, from smriti). This means:

- A file that moves or is copied keeps the same identity (same hash → same book).
- A file that is modified gets a new hash → a new book record. The old record and its segments remain intact for citation durability.
- Cross-edition identity (recognizing two different scans as "the same work") is out of scope for v1.

### segment

A **segment** is the universal decomposition unit — a format-native subdivision of a book:

| Format | Segment is |
|---|---|
| PDF | A page (rendered as image, or text if text layer exists) |
| epub | A chapter or section (from TOC/spine) |
| Markdown, plain text | A heading-delimited section, or fixed-size chunk if no structure |
| HTML | Heading-delimited sections following the document's hierarchy |

The embedding model (Qwen3-VL) handles both text and image segments in the same 2048-dim vector space. The retrieval interface is uniform regardless of source format.

### citation

A citation is the stable reference to a segment:

```
{book_id, segment_index, segment_label}
```

- `book_id` — the kosha book record UUID (derived from content_hash)
- `segment_index` — zero-based position in the decomposition sequence; stable across re-ingestion of the same file
- `segment_label` — human-friendly, best-effort, display-only: `"p.47"` for PDFs, `"ch.3 — Vibhuti Pada"` for epubs with TOC, `"§ Heading Name"` for text/HTML, `"seg.12"` as fallback

The stable key is `(book_id, segment_index)`. chitta stores the full triple so references remain meaningful even if kosha is offline.

## intake — what kosha ingests

kosha ingests all document types that constitute domain knowledge: PDFs, epubs, plain text, markdown, HTML, djvu, mobi, and other readable formats. The format determines the extraction pipeline, not whether kosha processes it.

### scoping

kosha has **configured roots** — directories that represent knowledge collections (e.g. `~/library/`, `~/notes/`, `~/research/`). Within those roots, it ingests files matching a configured set of extensions/mime types.

This is a kosha-side concept. smriti is unaware of it. kosha filters the event stream itself.

No kosha-side ignore mechanism for v1. Roots + mime filter is sufficient scoping. If a file is in a knowledge root and matches the filter, it gets ingested.

## how kosha learns about files

kosha subscribes to smriti's event stream via `smriti_events_since(cursor_id)`, a polling MCP tool over smriti's existing `events` table:

```
smriti scan completes
  → events persisted: {event_type, content_hash, path, mime_type, file_extension, ...}
  → kosha polls with cursor (last processed event id)
  → filters by path prefix (configured roots) + mime/extension
  → enqueues ingestion for created/updated events
  → updates source_path for moved/copied events (no re-ingestion, same hash)
  → marks books as gone for deleted events (segments retained for citation durability)
```

The event stream carries paths and metadata, not file bytes. When kosha is ready to ingest, it reads the file via `smriti_read`, going through smriti's privacy gate. This preserves:

- One audit point for all file reads
- One policy point (`.smritiignore` and root allowlists apply automatically)
- One identity point (BLAKE3 hash consistent across the ecosystem)

### cursor management

kosha tracks its own cursor (the last `events.id` it processed) in its database. smriti is stateless with respect to subscribers. If kosha falls behind for more than smriti's retention horizon (default 30 days via `smriti prune`), it loses events and must do a full reconciliation scan.

## embedding pipeline

### the hard problem is solvable

Most of the target corpus (astrology reference material) is scanned classical texts — no text layer, mixed scripts, complex layouts. **Qwen3-VL-Embedding-2B** encodes both text and images into the same 2048-dim vector space. A text query retrieves a scanned page directly. No OCR required.

### flow

For a document with a text layer (markdown, epub, text-layer PDF):

```
file read via smriti_read
  → decompose into text segments
  → each segment embedded via Qwen3-VL (text mode)
  → 2048-dim vector + full text stored
```

For a scanned PDF (no text layer):

```
file read via smriti_read
  → render each page to image
  → each page image embedded via Qwen3-VL (image mode)
  → 2048-dim vector stored, content_text is NULL
```

### model locality

The model runs locally. An upcoming benchmark will compare bge-m3 (dense+sparse, used by chitta) with Qwen3-VL. If Qwen3-VL matches or exceeds bge-m3 quality, the ecosystem may consolidate on one model. If not, two models coexist and the model server sketch (`model-server-sketch.md`) becomes load-bearing.

### error handling

If a segment fails to embed (corrupt page, OOM, etc.), kosha skips it and continues. A single bad segment does not block the rest of the book from being searchable. The book record tracks skipped segments. Status is `complete` with a caveat, not `failed`.

## process model

kosha is a single Rust binary with two workloads:

1. **Query serving** — MCP server, the primary role. Agents call `kosha_search`, `kosha_read`, etc.
2. **Background ingestion** — consumes the event stream, decomposes documents, generates embeddings. Runs as a background task pool with bounded concurrency.

Ingestion mostly happens offline (not while agents are actively querying), so contention is a non-issue in practice. If both run simultaneously, query serving takes priority.

## storage

**Postgres** with pgvector. Separate database from chitta (which also uses Postgres). The Postgres instance is already running in the ecosystem.

### schema

```sql
CREATE TABLE books (
    id              UUID PRIMARY KEY,
    content_hash    TEXT NOT NULL UNIQUE,
    source_path     TEXT NOT NULL,
    format          TEXT NOT NULL,        -- pdf, epub, markdown, txt, html, ...
    title           TEXT,
    segment_count   INTEGER DEFAULT 0,
    skipped_segments INTEGER DEFAULT 0,
    status          TEXT NOT NULL,        -- pending, ingesting, complete, failed
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);

CREATE TABLE segments (
    id              UUID PRIMARY KEY,
    book_id         UUID NOT NULL REFERENCES books(id),
    segment_index   INTEGER NOT NULL,
    segment_label   TEXT,
    content_text    TEXT,                 -- full text; NULL for image-only segments
    embedding       vector(2048),        -- Qwen3-VL vector
    created_at      TIMESTAMPTZ NOT NULL,
    UNIQUE(book_id, segment_index)
);

CREATE INDEX idx_segments_embedding ON segments
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE cursors (
    id              TEXT PRIMARY KEY,     -- e.g. 'smriti_events'
    last_event_id   INTEGER NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);
```

Full segment text is stored. This makes `kosha_read` fast (one query, no round-trip to smriti). The source file via `smriti_read` is the archival copy; kosha's text is the working copy for retrieval.

### retrieval

Vector search only for v1. `kosha_search` embeds the query with Qwen3-VL and searches `segments.embedding` via pgvector HNSW. Text-based FTS (tsvector over `content_text`) is a future consideration if exact-term retrieval proves weak.

## MCP tools

| Tool | Purpose |
|---|---|
| `kosha_search` | Semantic search across segments. Params: query, optional book filter, optional root filter, top_k. Returns: citations + snippets + similarity scores. |
| `kosha_read` | Fetch a segment by citation `(book_id, segment_index)`. Returns: full text or image description + metadata. |
| `kosha_book` | Metadata for a book: format, segment count, ingestion status, source path, content_hash. |
| `kosha_books` | List books, filterable by root, format, ingestion status. |
| `kosha_health` | Status check. |
| `kosha_ingest` | Manually trigger ingestion of a file (by path or hash), bypassing the event stream. For bootstrapping or one-offs. |

## query flows

**Discovery only** (smriti, not kosha):

```
agent  →  smriti_find "Bayesian inference"
             → paths + titles + topics for matching files
```

**Discovery → comprehension:**

```
agent  →  smriti_find "Brihat Parashara"
             → ~/library/classics/bphs.pdf
       →  kosha_search "Saturn Mars conjunction" book:<bphs-id>
             → segment citation + snippet + score
       →  kosha_read book:<bphs-id> segment:47
             → full text or image description
```

**Direct comprehension** (agent already knows the book):

```
agent  →  kosha_search "samadhi" book:<patanjali-id>
             → segment-level matches with citations
```

## what each side knows about the other

**smriti knows nothing about kosha.** It emits events and serves reads. No subscriber registry, no per-tool state.

**kosha knows two things about smriti:**

1. The `smriti_read` endpoint (for reading files through the privacy gate)
2. The `smriti_events_since` endpoint (for subscribing to scan events)

It also stores `content_hash` per book to correlate with smriti. That is the only schema-level coupling.

## file change handling

When a file changes, smriti emits an `updated` event with a new `content_hash`. kosha creates a new book record. The old book and its segments remain — chitta citations against the old book still resolve. No versioning machinery. New hash = new book.

For moves/copies (same hash, different path), kosha updates the `source_path` on the existing book record. No re-ingestion.

## deferred decisions

- **Segment size limits.** For large structureless text files, what's the max segment size before forcing a split? Deferred until the ingestion pipeline is built and we see real data.
- **Model version tracking.** If the embedding model is updated, do we re-embed everything? Schema for tracking model version per segment? Deferred.
- **FTS.** Text search (tsvector) alongside vector search, or reciprocal rank fusion. Deferred to post-v1.
- **OCR.** Text extraction from scanned pages. Deferred to post-v1; multimodal embedding handles retrieval without it.
- **Cross-edition identity.** Recognizing two different scans as the same work. Deferred; needs title-matching heuristics or user-supplied metadata.

## dependencies

| Dependency | Status | Needed for |
|---|---|---|
| smriti (event stream) | events table exists; `smriti_events_since` tool needed | file discovery, change notification |
| smriti (privacy gate) | `smriti_read` exists | file reads |
| Postgres + pgvector | available | storage + vector search |
| Qwen3-VL-Embedding-2B | available (Python); Rust via fastembed blocked on candle BF16 | text + image embedding |
| PDF rendering | available (poppler/mupdf) | page image generation |
| chitta | available | citation storage via tags |
