# Remote ingest (GPU pod → local DB)

Alternative to the dump-and-restore workflow in `cloud-ingest.md`. Instead of ingesting into the pod's database and transferring the dump, run kosha on the pod but point it at your local postgres over an SSH tunnel. Embedding runs on the pod's GPU; writes go directly to your local DB.

This avoids the dump/restore cycle entirely and makes incremental ingests trivial — spin up a pod, upload new files, ingest, done.

## Setup

### 1. Local postgres must accept TCP connections

Check `postgresql.conf`:

```
listen_addresses = 'localhost'   # default, fine for tunnel use
```

Check `pg_hba.conf` has a line allowing local TCP (it usually does by default):

```
host    kosha    your_user    127.0.0.1/32    md5
```

Reload if you changed anything: `sudo systemctl reload postgresql`.

### 2. Your home machine must be reachable from the pod

Options:

- **Tailscale** (easiest): both machines on the same tailnet. No port forwarding needed.
- **Port forwarding**: forward a port on your router to your machine's SSH port.
- **Reverse SSH tunnel from your machine to the pod**: connect outbound from home, establish the tunnel from that direction (avoids needing inbound access).

### 3. Open the tunnel

**If the pod can reach your machine** (tailscale or port forwarding):

```bash
# On the pod — forward pod's localhost:5432 to your machine's postgres
ssh -L 5432:localhost:5432 josh@your-home-ip
```

**If the pod can't reach your machine** (reverse tunnel from home):

```bash
# On your machine — connect to the pod, forward the pod's 5432 back to your local postgres
ssh -R 5432:localhost:5432 root@pod-ip -p <pod-ssh-port>
```

Either way, the pod's `localhost:5432` now reaches your local postgres.

### 4. Ingest from the pod

```bash
# On the pod:
DATABASE_URL=postgresql://your_user:your_pass@localhost/kosha \
  /workspace/kosha/target/release/kosha --device gpu ingest -r /workspace/corpus/ \
    --collection my-docs
```

Kosha embeds on the pod's GPU and writes chunks + embeddings through the tunnel to your local DB. Migrations run automatically on first connect.

## Performance considerations

- **Latency**: each chunk does a DB insert with an embedding vector (~4KB for 2048-dim halfvec). At typical home internet latencies (20-50ms RTT), this adds ~20-50ms per chunk on top of the embedding time. Embedding a chunk takes ~50-200ms on an L4, so the tunnel overhead roughly doubles wall-clock time in the worst case.
- **Bandwidth**: not a concern. A 2048-dim halfvec is ~4KB. Even at 10 chunks/sec that's 40KB/s.
- **Batch inserts**: if tunnel latency becomes the bottleneck, a future optimization would be batching inserts (insert N chunks per round-trip instead of 1). Not implemented yet.
- **Connection drops**: if the SSH tunnel drops mid-ingest, kosha will error on the next DB write. Re-establish the tunnel and re-run ingest — already-ingested files are skipped by content hash.

## When to use this vs. dump-and-restore

| Scenario | Recommendation |
|---|---|
| First big ingest, no local DB yet | Either works. Dump-and-restore is simpler. |
| Incremental ingest (adding new docs to existing DB) | Remote ingest. No need to upload/download the full DB. |
| Slow home internet (>100ms RTT or <10 Mbps) | Dump-and-restore. Tunnel latency will bottleneck inserts. |
| Large corpus, fast connection | Remote ingest. Saves the dump/transfer/restore cycle. |

## Future: merge tool

A `kosha merge` subcommand that imports from a dump file into an existing local DB, upserting by content hash. This would give the best of both worlds — ingest on the pod into a local pod DB (no tunnel latency), dump, transfer home, merge into your local DB without replacing it.

Rough design:

- Input: a `pg_dump -Fc` file or a running remote `DATABASE_URL`.
- For each leaf in the source: check if `content_hash` exists locally. If yes, skip. If no, copy the leaf, its segments, chunks (with embeddings), and tags.
- Collection and tag assignments could be merged or overwritten (flag to control).
- Needs careful ordering: leaves first, then segments, then chunks (foreign key deps).

Not yet implemented. For now, use remote ingest or full dump-and-restore.
