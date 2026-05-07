-- Collections + tags for scoped search.
-- Each leaf belongs to exactly one collection; tags refine within/across.

ALTER TABLE leaves ADD COLUMN collection TEXT NOT NULL DEFAULT 'default';
CREATE INDEX leaves_collection_idx ON leaves (collection);

CREATE TABLE leaf_tags (
    leaf_id  TEXT NOT NULL REFERENCES leaves(content_hash) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (leaf_id, tag)
);
CREATE INDEX leaf_tags_tag_idx ON leaf_tags (tag);
