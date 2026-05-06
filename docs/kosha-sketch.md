# kosha — sketch

Status: sketch
Date: 2026-04-27
Context: document intelligence layer for the aion ecosystem; replaces the earlier "grantha" concept. Sits on top of smriti.

## what it is

kosha (वेदकोश — "treasury of knowledge") is a document intelligence layer that sits on top of smriti. Smriti knows you have `brihat-parashara-hora-shastra.pdf` and where it lives. kosha knows what's on page 47.

It decomposes structured documents (PDFs, epubs) into pages and sections, extracts what content it can, embeds pages as vectors (text or image, same pipeline), and exposes per-page semantic search and citation tools over MCP.

## what it is not

- Not a filesystem tracker. Smriti does that. kosha subscribes to smriti for file events.
- Not a general-purpose search engine. It searches *within* documents that smriti already knows about.
- Not an OCR pipeline. Text extraction runs where a text layer exists. For scan-only pages, the page image is embedded directly — no OCR required.

## the hard problem is now solvable

Most astrology reference material is scanned classical texts — no text layer, mixed scripts (Sanskrit, transliterated, English commentary), complex layouts with tables and diagrams.

The original grantha sketch treated this as a deferred problem: "store images now, embed them later when good local multimodal models arrive." That day has arrived.

**Qwen3-VL-Embedding-2B** is a multimodal embedding model that encodes both text and images into the same 2048-dim vector space. A text query can retrieve a scanned page. No OCR. No two-phase architecture.

This collapses the v1/v2 staging from the grantha sketch. Page images can be embedded from day one — there is no "cataloged but not yet searchable" status for scan-only pages. Every page is searchable once embedded.

## embedding strategies

Two paths, both feed the same `page_vectors` table:

| strategy | input | model | speed (CPU) | use when |
|---|---|---|---|---|
| text embedding | extracted text | Qwen3-VL-Embedding-2B | ~1.4 s/page | text layer exists |
| image embedding | rendered page image | Qwen3-VL-Embedding-2B | ~253 s/page | scan-only page |

Image embedding on CPU is slow (253 s/page = ~4 min/page). A 500-page scan would take ~35 hours locally. This is where RunPod batch comes in.

### RunPod batch for image embedding

The practical path for bulk image embedding:

1. Render all pages to images locally (fast — poppler/mupdf)
2. Upload rendered images to a RunPod GPU instance
3. Run Qwen3-VL-Embedding-2B batch inference there
4. Download the populated embedding rows
5. Merge into local SQLite

Local text embedding stays local (1.4 s/page is fine). GPU offload is only needed for image embedding at scale.

## how it relates to the ecosystem

```
smriti (filesystem perception)
  │ "new PDF appeared at ~/library/classics/bphs.pdf"
  │ "file moved from ~/Downloads/"
  v
kosha (document intelligence)
  │ decompose into pages, extract text, render images
  │ embed via model server (text: fast, image: batch/GPU)
  │ expose citation + search tools
  v
chitta (memory)
  "research note about Saturn-Mars conjunction,
   cites book:bphs page 47"
```

- **smriti → kosha:** smriti tracks files, kosha processes them. kosha registers interest in configured file types (`.pdf`, `.epub`) within smriti roots. When smriti detects a new or changed document, kosha re-processes it.
- **kosha → chitta:** chitta memories reference kosha documents via `book:<id>` tags and `metadata.page`. kosha provides the citation; chitta stores the relationship.
- **kosha → model server:** kosha sends text or page images to the shared model server. Doesn't load models itself. See `model-server-sketch.md`.

## storage

SQLite, same single-file story as smriti. Separate DB (`~/.kosha/index.db`).

```sql
-- one row per document (linked to smriti by content_hash)
CREATE TABLE books (
    id TEXT PRIMARY KEY,             -- stable book id (slug or derived)
    content_hash TEXT NOT NULL,      -- smriti's content hash for this version
    title TEXT,
    author TEXT,
    format TEXT NOT NULL,            -- pdf, epub
    page_count INTEGER,
    has_text_layer BOOLEAN,
    extraction_status TEXT,          -- complete, partial, images_only
    first_indexed TIMESTAMP NOT NULL,
    last_indexed TIMESTAMP NOT NULL
);

-- one row per page
CREATE TABLE pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL REFERENCES books(id),
    page_number INTEGER NOT NULL,
    text_content TEXT,               -- extracted text, NULL if scan-only
    image_path TEXT,                 -- rendered page image (on-disk or blob)
    has_text BOOLEAN NOT NULL,
    embed_status TEXT NOT NULL DEFAULT 'pending',  -- pending, text, image, done
    UNIQUE(book_id, page_number)
);

-- per-page embeddings — 2048 dims for Qwen3-VL-Embedding-2B
CREATE VIRTUAL TABLE page_vectors USING vec0(
    page_id INTEGER PRIMARY KEY,
    embedding FLOAT[2048]
);

-- BM25 over extracted text (scan-only pages won't appear here)
CREATE VIRTUAL TABLE page_fts USING fts5(
    page_id UNINDEXED,
    book_id UNINDEXED,
    text_content
);
```

The `embed_status` column tracks per-page embedding state and makes the RunPod batch workflow safe to restart.

## MCP tools (sketch)

- **kosha_search** — semantic query across all books, returns page-level results with snippets and book/page citations
- **kosha_read** — read a page or page range from a book (returns text where available, describes image otherwise)
- **kosha_cite** — returns a citable reference (book id + page + snippet) formatted for chitta metadata
- **kosha_books** — list indexed books with extraction and embedding status
- **kosha_outline** — table of contents / structure of a book (where extractable)

## book identity

A book needs a stable `id` that survives re-scans, file moves, and new editions.

- **ISBN** where available (embedded in PDF metadata or epub)
- **Derived slug** from title + author (e.g. `bphs-parashara`) as fallback
- **Content hash** is *not* the id — it changes with the file version

The `book:<id>` tag in chitta references this id. If a new scan of the same book appears (different file, different hash), kosha links it to the same book id; chitta citations stay valid.

For now: human-assignable slugs with auto-suggestion from metadata. This needs more thought before implementation — wrong answer means broken citations.

## fastembed Rust integration (status)

The `fastembed` Rust crate (v5.13.3) has experimental `Qwen3VLEmbedding` support via the candle backend. This is the target for Rust-native integration (no Python subprocess, no ONNX detour).

Currently blocked:
- candle lacks BF16 CPU matrix multiply
- F16 dtype mismatches in fastembed's forward pass

Fix in progress upstream. When resolved, kosha can embed text natively in Rust at ~1.4 s/page. Image embedding will likely still go through Python/RunPod for the near term given the candle gaps.

## what it depends on

| dependency | status | needed for |
|---|---|---|
| smriti v0.1 | complete | file awareness, change events |
| model server | sketch | embeddings (see model-server-sketch.md) |
| Qwen3-VL-Embedding-2B | available (Python) | text + image embedding |
| fastembed Qwen3VL (Rust) | blocked on candle BF16 | native Rust embedding |
| PDF text extraction | available (poppler, lopdf) | text layer extraction |
| PDF/image rendering | available (mupdf, poppler) | page image generation for GPU batch |
| chitta | available | citation storage via tags |

## open questions

- **Page image storage.** Files on disk (referenced by path) or blobs in SQLite? Files are simpler and work better for RunPod upload, but add a second artifact to manage.
- **Chunking strategy.** Pages are a natural unit for PDFs. For epubs, chapters/sections might be better. Both: pages for citation anchoring, overlapping chunks for embedding quality.
- **Hybrid search.** BM25 + dense (RRF merge) like smriti? BM25 only covers pages with text layers, so for scan-heavy collections dense search will dominate. Still worth having both.
- **Scope.** PDFs only first, then epubs? Epub extraction is much cleaner and probably worth doing early.
- **RunPod batch UX.** Who triggers the upload? Manual CLI command? Daemon that queues pending image embeds and waits for GPU availability?
- **Relation to smriti tiers.** Tier 1 in smriti (file-level) + richer in kosha (page-level). Two layers. Smriti stays unaware of page structure.
