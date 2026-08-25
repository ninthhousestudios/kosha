#!/usr/bin/env bash
# Dump the locally-ingested `kosha_adityas` corpus DB for restore on the VPS.
#
# The adityas being-corpus is already ingested locally with pre-computed
# embeddings (onnx NomicEmbedTextV15, 768-dim `halfvec`). Restoring this dump on
# the VPS avoids re-uploading the source corpus and re-running the embedder at
# ingest time — only query-time embedding runs on the VPS. See adityas/ai/36.
#
# The restore target MUST have the pgvector extension available: the dump emits
# `CREATE EXTENSION vector` and uses the `halfvec` type + an hnsw index. Use a
# pgvector Postgres image (e.g. pgvector/pgvector:pg17), not stock postgres.
set -euo pipefail

SRC_DB="${1:-kosha_adityas}"
OUT="${2:-kosha_adityas.dump}"

# Custom format (-Fc): compressed, restored with pg_restore. Carries the
# CREATE EXTENSION statement and the _sqlx_migrations table, so kosha's
# run_migrations() sees the schema as already-migrated and no-ops on boot.
pg_dump --format=custom --no-owner --no-privileges \
    --dbname="$SRC_DB" --file="$OUT"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
cat <<'EOF'

Restore on the VPS (into a pgvector Postgres, DB e.g. 'kosha'):
  createdb -U postgres kosha
  pg_restore --no-owner --role=kosha --dbname=kosha kosha_adityas.dump

Then point the kosha container at it:
  DATABASE_URL=postgres://kosha:<pw>@<pg-host>/kosha
  KOSHA_EMBED_PROVIDER=onnx  KOSHA_EMBED_MODEL=NomicEmbedTextV15
EOF
