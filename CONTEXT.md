# kosha

Document intelligence layer for the manas ecosystem. Decomposes documents into searchable segments, embeds them into a shared vector space, and exposes semantic search and citation tools over MCP.

## Language

**Leaf**:
A document ingested by kosha — a PDF, epub, markdown file, HTML page, or any readable file. One file = one leaf. The treasury (kosha) holds leaves.
_Avoid_: book, document, source, item

**Segment**:
A format-native subdivision of a leaf — a page (PDF), chapter (epub), heading-delimited section (markdown/HTML). Carries a human-meaningful label ("p. 47", "ch. 3 — Vibhuti Pada"). Structural and navigational, not the embedding unit.
_Avoid_: page (too format-specific), section (ambiguous)

**Chunk**:
A structurally-coherent subdivision of a segment, target-sized for embedding quality and bounded for retrieval predictability. Boundary candidates respect document structure (headings, code fences, paragraph breaks) within a configurable target ± tolerance window. The unit that gets embedded, searched against, and cited. A short segment may be a single chunk.
_Avoid_: block, fragment, window

**Citation**:
A stable reference to a chunk: `(leaf_id, segment_index, chunk_index)`. Inherits the parent segment's label; multi-chunk segments append a disambiguator: `"ch. 3 — Vibhuti Pada [2/7]"`.
_Avoid_: reference, pointer, link

## Relationships

- A **Leaf** is decomposed into one or more **Segments**
- A **Segment** is split into one or more **Chunks**
- A **Chunk** belongs to exactly one **Segment**, which belongs to exactly one **Leaf**
- A **Citation** points to exactly one **Chunk**
- chitta memories reference kosha content via **Citations**

## Example dialogue

> **Dev:** "When an agent searches kosha, what comes back?"
> **Domain expert:** "A **Chunk** — the matching passage with a **Citation**. If the agent needs more context, it can request a range of **Chunks** within the same **Segment**, or the whole **Segment**."

## Flagged ambiguities

- "book" was used in earlier sketches to mean **Leaf**. Resolved: **Leaf** is the canonical term.
- "segment" was initially the embedding unit. Resolved: **Segment** is structural; **Chunk** is the embedding unit.
- Chunk overlap: configurable, starting at zero. **Segments** hold authoritative text; **Chunks** are derived views. Overlap may be added later based on benchmarking with the actual corpus.
- Chunk size: 512 tokens default, configurable. Typical PDF text page (~300-400 tokens) stays as one chunk; epub chapters get split.
