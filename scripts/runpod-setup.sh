#!/usr/bin/env bash
set -euo pipefail

# RunPod setup script for kosha GPU ingest.
#
# Recommended pod:
#   Image: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#   GPU:   L40S or L4 (Qwen3-VL-2B fits easily)
#   vCPU:  12+
#   Disk:  40GB+ (postgres WAL can bloat under bulk inserts)
#   RAM:   32GB+
#
# Do NOT use:
#   - Docker Hub images (rate-limited on RunPod)
#   - Ubuntu 20.04 (ships PG 12, pgvector needs 13+)
#
# Usage:
#   git clone <your-kosha-repo> /workspace/kosha
#   bash /workspace/kosha/scripts/runpod-setup.sh
#
# After setup, ingest your corpus:
#   /workspace/kosha/target/release/kosha --device gpu ingest -r /workspace/corpus/
#
# Then dump and transfer home:
#   pg_dump -U postgres -Fc kosha > /workspace/kosha-dump.pg
#   runpodctl send /workspace/kosha-dump.pg

KOSHA_DIR="${KOSHA_DIR:-/workspace/kosha}"

echo "=== kosha RunPod setup ==="
echo "kosha dir: $KOSHA_DIR"

# ── System deps ─────────────────────────────────────────────────────
# poppler + cairo are needed for PDF decomposition (poppler-rs, cairo-rs)
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    postgresql postgresql-contrib postgresql-server-dev-all \
    build-essential git curl pkg-config libssl-dev \
    libpoppler-glib-dev libcairo2-dev

# ── Rust toolchain ──────────────────────────────────────────────────
# RunPod images ship old Rust (~1.75). kosha uses edition 2024, needs 1.85+.
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"
echo "Updating Rust toolchain..."
rustup update stable

echo "Rust: $(rustc --version)"

# ── pgvector extension ──────────────────────────────────────────────
PG_SHAREDIR=$(pg_config --sharedir 2>/dev/null || echo "")
if [ -n "$PG_SHAREDIR" ] && [ -f "$PG_SHAREDIR/extension/vector.control" ]; then
    echo "pgvector already installed."
else
    echo "Building pgvector from source..."
    rm -rf /tmp/pgvector
    git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector
    cd /tmp/pgvector
    make && make install
    cd /
fi

# ── Start postgres ──────────────────────────────────────────────────
pg_isready -q 2>/dev/null || pg_ctlcluster $(pg_lsclusters -h | awk '{print $1, $2}') start
sleep 2

# ── Postgres setup ──────────────────────────────────────────────────
# Default auth is peer over unix sockets. We set a password so kosha
# can connect via TCP with DATABASE_URL. Don't change pg_hba.conf
# peer→md5 for the postgres OS user — that breaks su - postgres.
su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres';\""

su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='kosha'\"" | grep -q 1 || \
    su - postgres -c "createdb kosha"

su - postgres -c "psql -d kosha -c 'CREATE EXTENSION IF NOT EXISTS vector;'"

# Bump WAL size limit — bulk inserts generate a lot of WAL.
su - postgres -c "psql -c \"ALTER SYSTEM SET max_wal_size = '2GB';\""
su - postgres -c "psql -c 'SELECT pg_reload_conf();'"

echo "Postgres ready: kosha database with pgvector"

# ── Build kosha ─────────────────────────────────────────────────────
cd "$KOSHA_DIR"

cat > .env <<'EOF'
DATABASE_URL=postgresql://postgres:postgres@localhost/kosha
EOF

echo "Building kosha with CUDA support (this takes ~5-10 min on first run)..."
cargo build --release --features cuda
echo "Built: $KOSHA_DIR/target/release/kosha"

# ── Verify ──────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify GPU detection:"
echo "  $KOSHA_DIR/target/release/kosha --device gpu ingest --help"
echo ""
echo "Ingest a corpus:"
echo "  $KOSHA_DIR/target/release/kosha --device gpu ingest -r /workspace/corpus/ --collection my-docs"
echo ""
echo "After ingest, dump the database:"
echo "  pg_dump -U postgres -Fc kosha > /workspace/kosha-dump.pg"
echo ""
echo "Transfer home:"
echo "  runpodctl send /workspace/kosha-dump.pg"
echo "  # or: scp -P <port> root@<pod-ip>:/workspace/kosha-dump.pg ."
echo ""
echo "Post-ingest cleanup (reclaim disk from WAL bloat):"
echo "  su - postgres -c \"psql -d kosha -c 'VACUUM FULL; CHECKPOINT;'\""
