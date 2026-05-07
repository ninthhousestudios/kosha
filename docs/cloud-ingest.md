# Cloud ingest workflow (RunPod)

Run `kosha ingest` on a GPU instance, dump the database, restore locally. Embedding is the bottleneck — GPU makes it 10-50x faster.

## Pod configuration

- **Image**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **GPU**: L40S (48GB VRAM) or L4 — the Qwen3-VL-2B model fits easily
- **vCPU**: 12+
- **RAM**: 32GB+
- **Disk**: 40GB+ (postgres WAL bloat from bulk inserts can eat 10-20GB)

Do NOT use Docker Hub images (rate-limited on RunPod) or Ubuntu 20.04 (ships PG 12, pgvector needs 13+).

## Setup script

```bash
#!/usr/bin/env bash
set -euo pipefail

# ── System deps ─────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq postgresql postgresql-contrib postgresql-server-dev-all \
    build-essential git curl pkg-config libssl-dev \
    libpoppler-glib-dev libcairo2-dev

# ── Rust toolchain ──────────────────────────────────────────────────
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"
# RunPod images ship old Rust; edition2024 needs 1.85+
rustup update stable

# ── pgvector ────────────────────────────────────────────────────────
PG_SHAREDIR=$(pg_config --sharedir 2>/dev/null || echo "")
if [ -n "$PG_SHAREDIR" ] && [ -f "$PG_SHAREDIR/extension/vector.control" ]; then
    echo "pgvector already installed"
else
    cd /tmp
    git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git
    cd pgvector && make && make install
fi

# ── Start postgres ──────────────────────────────────────────────────
pg_isready -q 2>/dev/null || pg_ctlcluster $(pg_lsclusters -h | awk '{print $1, $2}') start
sleep 2

# ── Postgres setup ──────────────────────────────────────────────────
su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres';\""
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='kosha'\"" | grep -q 1 || \
    su - postgres -c "createdb kosha"
su - postgres -c "psql -d kosha -c 'CREATE EXTENSION IF NOT EXISTS vector;'"
su - postgres -c "psql -c \"ALTER SYSTEM SET max_wal_size = '2GB';\""
su - postgres -c "psql -c 'SELECT pg_reload_conf();'"
echo "Postgres ready: kosha with pgvector"

# ── Clone and build kosha ──────────────────────────────────────────
WORK_DIR="${WORK_DIR:-/workspace}"
cd "$WORK_DIR"
if [ ! -d kosha ]; then
    git clone <your-kosha-repo-url> kosha
fi
cd kosha

cat > .env <<'EOF'
DATABASE_URL=postgresql://postgres:postgres@localhost/kosha
EOF

echo "Building kosha with CUDA support..."
cargo build --release --features cuda
echo "Built: $WORK_DIR/kosha/target/release/kosha"
```

## Running ingest

```bash
cd /workspace/kosha

# Copy your corpus to the pod first (rsync, scp, runpodctl send, etc.)
# Then ingest:
./target/release/kosha --device gpu ingest -r /workspace/corpus/ \
    --collection my-docs --tag batch-2026-05

# The --device flag:
#   gpu  — use CUDA (fails if no GPU or not compiled with cuda feature)
#   cpu  — force CPU
#   auto — try CUDA, fall back to CPU (default)
```

## Dumping and restoring

```bash
# On the pod: dump the kosha database
pg_dump -U postgres -Fc kosha > /workspace/kosha-dump.pg

# Transfer to local machine
# Option A: runpodctl
runpodctl send /workspace/kosha-dump.pg
# Option B: scp
scp -P <port> root@<pod-ip>:/workspace/kosha-dump.pg .

# Locally: restore into your kosha database
pg_restore -U <user> -d kosha --clean --if-exists kosha-dump.pg
# Or create fresh:
createdb kosha_cloud
psql -d kosha_cloud -c 'CREATE EXTENSION IF NOT EXISTS vector;'
pg_restore -U <user> -d kosha_cloud kosha-dump.pg
```

## Known gotchas

1. **Rust version**: RunPod images ship old Rust (1.75ish). Kosha uses edition 2024 which needs 1.85+. Always `rustup update stable` first.

2. **Disk bloat**: Postgres WAL + dead tuples from bulk inserts. The `max_wal_size = '2GB'` setting helps. After large ingests, run `VACUUM FULL` and `CHECKPOINT`.

3. **OpenBLAS thread bomb**: If anything pulls in numpy/scipy (unlikely for kosha, but if debugging with Python tools), set `OPENBLAS_NUM_THREADS=4` or it tries 64 threads and crashes on limited-vCPU pods.

4. **Postgres auth**: Default is peer auth over unix sockets. The setup script uses `ALTER USER` + password in DATABASE_URL for TCP connections. If you get auth errors, check pg_hba.conf — changing peer to md5 for the postgres OS user breaks `su - postgres` commands.

5. **Docker Hub rate limiting**: Don't try to pull images from Docker Hub on RunPod. Use RunPod's own base images.

6. **libpoppler / libcairo**: Required for PDF decomposition. The `apt-get install libpoppler-glib-dev libcairo2-dev` line in setup covers this. Without them, `cargo build` fails on poppler-sys/cairo-sys.

7. **CUDA + candle**: The `--features cuda` flag enables candle's CUDA backend. This pulls in CUDA kernel compilation at build time, which needs the CUDA toolkit — the recommended RunPod image includes it. Build time is ~5-10 min on first run, cached after.

## Cost estimates

| Corpus size | Estimated GPU time | Pod cost (L40S ~$0.70/hr) |
|---|---|---|
| 100 documents | ~5-10 min | ~$0.10 |
| 1000 documents | ~30-60 min | ~$0.50 |
| 10000 documents | ~4-8 hours | ~$4-6 |

GPU time depends heavily on document sizes and segment counts. PDFs with many pages take longer than plain markdown files.
