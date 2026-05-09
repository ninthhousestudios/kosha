# Cloud ingest how-to (RunPod)

Run `kosha ingest` on a GPU instance, dump the database, restore locally. Embedding is the bottleneck — GPU makes it 10-50x faster than CPU.

## Step 1: Create the pod

Go to [runpod.io](https://runpod.io) → Pods → Deploy.

**Pod settings:**

| Setting | Value | Why |
|---|---|---|
| Image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` | Has CUDA toolkit (needed for candle GPU build). The `-devel` variant includes nvcc and headers. |
| GPU | L40S (48GB) or L4 (24GB) | Qwen3-VL-2B fits easily on either. L40S is faster. |
| vCPU | 12+ | Rust compilation is CPU-bound. More cores = faster build. |
| RAM | 32GB+ | Postgres + model loading. |
| Disk | 40GB+ | OS ~15GB, Rust build cache ~5GB, postgres data + WAL ~10-15GB, your corpus. |

Do NOT use Docker Hub images (rate-limited on RunPod) or Ubuntu 20.04 (ships PG 12, pgvector requires 13+).

## Step 2: Upload your corpus to the pod

Once the pod is running, open a terminal (web terminal or SSH).

**Option A: runpodctl (best for large files)**

```bash
# On your local machine — install runpodctl if you haven't:
# https://github.com/runpod/runpodctl

# Tar up your corpus
tar czf corpus.tar.gz ~/library/

# Send to the pod (gives you a code to receive on the other end)
runpodctl send corpus.tar.gz
```

```bash
# On the pod:
runpodctl receive <code>
mkdir -p /workspace/corpus
tar xzf corpus.tar.gz -C /workspace/corpus/
```

**Option B: scp**

Find SSH details in the RunPod pod page (Connect → SSH).

```bash
# From your local machine:
scp -P <port> -r ~/library/ root@<pod-ip>:/workspace/corpus/
```

**Option C: git (if your corpus is in a repo)**

```bash
# On the pod:
git clone <your-corpus-repo> /workspace/corpus
```

## Step 3: Clone kosha and run setup

```bash
# On the pod:
git clone <your-kosha-repo-url> /workspace/kosha
bash /workspace/kosha/scripts/runpod-setup.sh
```

The setup script handles everything: system deps (postgres, pgvector, poppler, cairo), Rust toolchain update, database creation, and `cargo build --release --features cuda`. First build takes ~5-10 minutes.

If you need to re-run setup (e.g. pod was stopped and restarted), the script is idempotent — it skips steps that are already done.

## Step 4: Ingest

```bash
cd /workspace/kosha

# Ingest a directory recursively
./target/release/kosha --device gpu ingest -r /workspace/corpus/ \
    --collection my-docs --tag batch-2026-05

# Or ingest specific files
./target/release/kosha --device gpu ingest paper.pdf thesis.epub

# The --device flag:
#   gpu  — force CUDA (fails if no GPU available)
#   cpu  — force CPU
#   auto — try CUDA, fall back to CPU (default)
```

kosha prints progress per file: segment and chunk counts, or "skipped" for files already ingested (same BLAKE3 hash). You can safely re-run ingest if it gets interrupted.

## Step 5: Dump the database

```bash
# On the pod:
pg_dump -U postgres -Fc kosha > /workspace/kosha-dump.pg

# Check the size — if it's unexpectedly large, vacuum first:
su - postgres -c "psql -d kosha -c 'VACUUM FULL; CHECKPOINT;'"
pg_dump -U postgres -Fc kosha > /workspace/kosha-dump.pg
```

## Step 6: Download the dump

**Option A: runpodctl**

```bash
# On the pod:
runpodctl send /workspace/kosha-dump.pg

# On your local machine:
runpodctl receive <code>
```

**Option B: scp**

```bash
# From your local machine:
scp -P <port> root@<pod-ip>:/workspace/kosha-dump.pg .
```

## Step 7: Restore locally

```bash
# Into an existing kosha database (replaces contents):
pg_restore -U <user> -d kosha --clean --if-exists kosha-dump.pg

# Or into a fresh database:
createdb kosha
psql -d kosha -c 'CREATE EXTENSION IF NOT EXISTS vector;'
pg_restore -U <user> -d kosha kosha-dump.pg
```

Verify it worked:

```bash
export DATABASE_URL="postgresql://<user>:<pass>@localhost/kosha"
kosha list
```

## Step 8: Terminate the pod

Don't forget — RunPod charges by the hour. Once you've downloaded the dump, terminate the pod.

## Known gotchas

1. **Rust version**: RunPod images ship old Rust (~1.75). kosha uses edition 2024 which needs 1.85+. The setup script runs `rustup update stable` to handle this.

2. **Disk bloat**: Postgres WAL + dead tuples from bulk inserts. The setup script sets `max_wal_size = '2GB'`. After large ingests, `VACUUM FULL` + `CHECKPOINT` reclaims space. A 1000-document ingest might temporarily use 15-20GB of disk for WAL alone.

3. **Postgres auth**: Default is peer auth over unix sockets. The setup script sets a password for TCP connections via `DATABASE_URL`. Don't change pg_hba.conf `peer → md5` for the postgres OS user — that breaks `su - postgres` commands that the setup script uses.

4. **Docker Hub rate limiting**: RunPod can't pull from Docker Hub reliably. Always use RunPod's own base images (the `runpod/pytorch:*` images).

5. **libpoppler / libcairo**: Required for PDF decomposition. Without them, `cargo build` fails on poppler-sys / cairo-sys. The setup script installs both.

6. **CUDA build time**: `--features cuda` pulls in candle CUDA kernel compilation, which needs nvcc from the CUDA toolkit. The recommended RunPod image (`-devel` variant) includes this. First build is ~5-10 min; subsequent builds are fast (cargo cache).

7. **Model download**: First ingest downloads the Qwen3-VL-2B model from HuggingFace (~4GB). This happens automatically but takes a few minutes on first run.

8. **OpenBLAS thread bomb**: If you install any Python tools that pull in numpy/scipy for debugging, set `OPENBLAS_NUM_THREADS=4` first, or it tries to spawn 64 threads and crashes on 12-vCPU pods.

## Cost estimates

| Corpus | Estimated GPU time | Pod cost (L40S ~$0.70/hr) |
|---|---|---|
| 100 documents | ~5-10 min | ~$0.10 |
| 1000 documents | ~30-60 min | ~$0.50 |
| 10000 documents | ~4-8 hours | ~$4-6 |

Times vary with document size. A 500-page PDF has 500 segments to embed; a markdown file might have 5. Budget for the upper end.
