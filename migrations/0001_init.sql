-- kosha v0.1.0 initial schema.
-- Three-level hierarchy: leaves → segments → chunks.
-- Leaf keyed by BLAKE3 content hash for deterministic, content-addressed identity.

create extension if not exists vector;

create table leaves (
    content_hash  text         primary key,
    source_path   text         not null,
    format        text         not null,
    title         text,
    segment_count integer      not null default 0,
    status        text         not null default 'pending',
    error         text,
    created_at    timestamptz  not null default now(),
    updated_at    timestamptz  not null default now()
);

create index leaves_status_idx on leaves (status);

create table segments (
    id             uuid         primary key,
    leaf_id        text         not null references leaves(content_hash),
    segment_index  integer      not null,
    segment_label  text         not null,
    content_text   text,
    created_at     timestamptz  not null default now(),
    unique (leaf_id, segment_index)
);

create index segments_leaf_id_idx on segments (leaf_id);

create table chunks (
    id              uuid           primary key,
    segment_id      uuid           not null references segments(id),
    chunk_index     integer        not null,
    leaf_id         text           not null,
    segment_index   integer        not null,
    chunk_label     text           not null,
    content_text    text,
    embedding       halfvec(2048),
    created_at      timestamptz    not null default now(),
    unique (segment_id, chunk_index)
);

-- Denormalized leaf_id + segment_index for join-free citation assembly.
create index chunks_citation_idx on chunks (leaf_id, segment_index, chunk_index);

-- ANN search on embeddings. HNSW with cosine distance.
-- halfvec supports up to 4000 dims for HNSW (vector caps at 2000).
create index chunks_embedding_idx
    on chunks using hnsw (embedding halfvec_cosine_ops)
    with (m = 16, ef_construction = 64);
