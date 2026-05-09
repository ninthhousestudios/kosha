# kosha — handoff

Date: 2026-05-07

## Pick up next

### 1. Retag / move command

Currently, re-ingesting an already-ready leaf skips it — so you can't change collection or tags after initial ingest. Need either:
- A `kosha retag <hash> --collection <name> --tag <tag>` CLI command
- Or an `kosha_retag` MCP tool
This is just store-level UPDATE + set_leaf_tags, no re-embedding needed.

### 2. Format decomposers — epub, markdown/HTML (kosha/14, kosha/15)

The Decomposer trait takes `&[u8]` and returns `SegmentContent::Text` or `SegmentContent::Image`. File extension dispatch is in `ingest.rs::decomposer_for()`.

### 3. Hybrid search (kosha/22)

Lexical leg + RRF fusion. Currently search is vector-only.

### 4. HttpEmbedder image support

The `embed_image_bytes` method returns "not supported" for the HTTP provider. Low priority — local embedder works.

## Task graph

PRD is kosha/7. Done: 8, 9, 10, 11, 12, 13, 20, 23. Remaining: 14 (epub), 15 (md/html), 16 (kosha_ingest MCP tool), 17 (directory ingest), 18 (systemd unit), 21 (LLM cache), 22 (hybrid search), 24 (explain mode), 25 (graceful tiering doc).

Collections + tags feature is complete and verified (commit 70e7a1c) but not yet tracked in yojana.

## Context the next session needs

- Collections: single collection per leaf (TEXT column, default 'default'), tags via `leaf_tags` join table
- Search/list accept `collections: Option<Vec<String>>` and `tags: Option<Vec<String>>`
- New `kosha_collections` MCP tool lists distinct collection names
- CLI: `kosha ingest <path> --collection <name> --tag <tag>` (tag is repeatable)
- DB has test data: 5 leaves across 4 collections (astro, biz, default, yoga)
- `.kosha/.env` now exists at `~/.kosha/.env` with `DATABASE_URL=postgres://josh@localhost/kosha`
