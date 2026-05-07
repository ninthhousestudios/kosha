-- Track which provider/model produced each chunk's embedding.
-- Defaults match the only provider that existed before this migration.
ALTER TABLE chunks ADD COLUMN embed_provider  TEXT    NOT NULL DEFAULT 'local';
ALTER TABLE chunks ADD COLUMN embed_model     TEXT    NOT NULL DEFAULT 'Qwen/Qwen3-VL-Embedding-2B';
ALTER TABLE chunks ADD COLUMN embed_dimension INTEGER NOT NULL DEFAULT 2048;
