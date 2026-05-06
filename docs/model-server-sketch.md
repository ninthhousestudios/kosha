# model server — sketch

Status: sketch
Date: 2026-04-27
Context: multiple tools in Josh's ecosystem need ML models but the laptop has only 14GB RAM. This sketches a shared model server to manage that constraint.

## the problem

Several tools need embedding models, and they can't all hold models in RAM at once:

| tool | model | size | pattern |
|---|---|---|---|
| smriti | BGE-M3 (ONNX) | ~1.1 GB | on-demand per search query |
| vedakosha | Qwen3-VL-Embedding-2B | 604 MB resident, 4.6 GB peak during image embed | sustained during indexing |
| chitta | BGE-M3 (ONNX) | ~1.1 GB | on-demand per memory search |
| panda | all-MiniLM-L6-v2 (fastembed/ONNX) | ~90 MB | transient, loads per invocation |
| aion (future) | TBD | TBD | TBD |

Hardware: 14 GB RAM total, ~9 GB available. AMD Radeon 680M integrated GPU — no CUDA, no ROCm support worth relying on.

**The constraint is simple:** only one large model can be resident at a time. BGE-M3 and Qwen3-VL-Embedding-2B cannot coexist.

Panda's MiniLM is small enough (~90 MB) and transient enough (loads/unloads per invocation) that it is not a problem — it can stay independent of the server.

## what the model server does

A single long-running process that:

1. Manages model lifecycle — loads a model on first request, holds it resident while in use, unloads when idle or when a different model is requested
2. Accepts embedding requests over IPC (unix socket)
3. Returns embedding vectors to the caller
4. Enforces the one-large-model-at-a-time constraint — serializes load/unload, queues requests if a different model is being loaded

Tools (smriti, vedakosha, chitta) become thin clients. They send text or images over the socket and receive vectors back. No model code in the tool processes.

## consolidation question

Qwen3-VL-Embedding-2B handles both text and images and produces 2048-dim vectors. BGE-M3 produces 1024-dim vectors and is text-only.

**Option A — single model (Qwen3-VL-Embedding-2B for everything):**
- One model to manage. Server simplicity.
- 2048-dim vectors are double the storage. `document_vectors` tables in smriti and chitta would need migration.
- Search cost scales with dimension. For smriti's use case (file-level BM25+dense hybrid), the extra dims may not be worth it.
- Qwen3-VL text embedding quality vs BGE-M3 is untested in this ecosystem.

**Option B — multiple models, one resident at a time:**
- smriti and chitta keep BGE-M3 (no migration needed)
- vedakosha uses Qwen3-VL-Embedding-2B
- Server switches between them based on which tool is active
- Switching cost: unload old, load new (a few seconds). Acceptable if not interleaved constantly.
- Tool isolation: smriti search results and vedakosha search results are not cross-comparable (different embedding spaces)

Option B is lower risk for a first implementation. Option A becomes attractive if Qwen3-VL text quality proves comparable to BGE-M3 and the migration cost is paid once.

Another question: which is better for smriti. Also, I want to review the purpose of
embeddings in smriti. Vedakosha sits on top of smriti, so both need embeddings? Just
want to make sure this makes sense.

## IPC protocol

Three candidates:

**Unix socket + simple framing (recommended for v1)**
- Low latency, no network stack, no auth complexity
- Framing: 4-byte length prefix + msgpack or JSON body
- Works well for local IPC; no external dependencies
- Adequate for the request rates here (not a high-throughput scenario)

**gRPC**
- Good for structured schemas and streaming
- Heavier dependency (tonic, protobuf codegen)
- Overkill for a single-machine local server

**HTTP (localhost)**
- Simplest to debug (curl works)
- Slightly more overhead than unix sockets for local IPC
- Reasonable option if gRPC feels too heavy

Recommendation: unix socket + JSON for v1. Migrate to gRPC if the protocol needs versioning or streaming becomes important.

## RunPod batch — separate concern

RunPod GPU offload for vedakosha image embedding is a *batch job*, not a server request. It doesn't fit neatly into the model server architecture:

- Local model server handles: real-time text embedding (1.4 s/page, CPU), low-volume image embedding (slow but works for small jobs)
- RunPod handles: bulk image embedding (hundreds of scanned pages), one-off batch jobs triggered manually or by CLI

The split: the model server serves the hot path (text embedding, small image jobs). RunPod serves the cold path (initial indexing of large scan collections).

The RunPod batch workflow is a separate CLI command (`vedakosha batch-embed --runpod`), not a model server feature. The server doesn't know about RunPod.

## model lifecycle

```
[ client sends embed request ]
        |
        v
[ server checks: is the right model loaded? ]
    YES --> forward request, return vectors
    NO  --> unload current model (if any)
            load requested model
            forward request, return vectors
```

Idle timeout: unload a model after N minutes of inactivity (configurable). Keeps RAM free when nothing is running.

Request queuing: if a model switch is in progress, queue incoming requests rather than failing them. Switch takes a few seconds; callers can wait.

## socket protocol sketch

Request:
```json
{
  "model": "qwen3-vl-2b",
  "input_type": "text",
  "content": "Saturn in the 7th house...",
  "batch": false
}
```

Image request:
```json
{
  "model": "qwen3-vl-2b",
  "input_type": "image",
  "image_path": "/path/to/page-0047.png",
  "batch": false
}
```

Batch request:
```json
{
  "model": "qwen3-vl-2b",
  "input_type": "text",
  "batch": true,
  "items": ["text 1", "text 2", "..."]
}
```

Response:
```json
{
  "ok": true,
  "embedding": [0.012, -0.034, ...],
  "dims": 2048,
  "model": "qwen3-vl-2b"
}
```

Errors:
```json
{
  "ok": false,
  "error": "model_load_failed",
  "message": "insufficient memory to load qwen3-vl-2b"
}
```

## implementation path

v1 — text embedding only, single model:
- Daemon process, unix socket listener
- Loads BGE-M3 or Qwen3-VL (configured per deployment)
- JSON framing, synchronous requests
- Idle timeout + graceful shutdown
- smriti and chitta updated to use the socket instead of loading ONNX directly

v2 — multi-model support:
- Model registry (name → load function)
- Explicit load/unload RPC
- Request routing by `model` field
- Metrics: queue depth, load time, resident model

v3 — image embedding:
- Image input type in the protocol
- Depends on fastembed candle BF16 fix or Python subprocess fallback

## open questions

- **Do we consolidate onto Qwen3-VL-Embedding-2B?** The dimension change (1024 → 2048) requires migrating smriti and chitta vector tables. Storage doubles. Worth it only if Qwen3-VL text quality is clearly better, or if the single-model simplicity pays for itself.
- we are designing an astrology-benchmark for chitta so we can test this. especially the
sparse vectors how much that really matters.
- **Can panda stay independent?** Yes, almost certainly. MiniLM is 90 MB and transient. Folding it into the server adds complexity for no benefit. Leave it alone. Agreed.
- **gRPC vs unix socket vs HTTP?** Unix socket + JSON is the right call for v1. Revisit if streaming or versioning become requirements. Yes
- **How does RunPod batch fit in?** It doesn't — it's a separate CLI workflow, not a model server feature. Server handles local real-time requests; RunPod handles offline bulk jobs.
- **Model hot-swapping frequency.** If smriti searches and vedakosha indexing are never interleaved in practice, the switching cost is irrelevant. If they are, the queue needs a priority mechanism (interactive search > background indexing). Needs clarity on the roles of the two.
- **Startup latency.** If the server unloads on idle, the first request after idle pays the load cost. Is that acceptable? Alternative: keep the model resident always (accept the RAM cost). Config knob.
- **Who owns the server process?** systemd user unit? smriti daemon spawns it? Self-starting (first client forks it)? Self-starting is simplest for development; systemd unit is better for production. Systemd. Server is its own package withs its own name. Another name! So many packages!


## Other things to consider

Beyond the core load/unload logic, there are several "engineering friction" points that
often emerge in these kinds of low-resource daemon setups.

  1. The "Thrashing" Problem & Sticky Models The sketch mentions an idle timeout, but
     doesn't explicitly address Model Thrashing. 
   * The Risk: If Tool A (Smriti) and Tool B (Vedakosha) are both running background
     tasks, the server might spend more time loading/unloading than actually doing
inference.
   * Improvement: Implement a Minimum Residency Time (e.g., 30 seconds). If a model is
     loaded, it must stay loaded for at least 30 seconds even if a different model is
requested. This prevents a "ping-pong" effect where the server dies trying to satisfy
two interleaved requests.

  2. Zero-Copy & Shared Memory (MMAP) Since you are on Linux and using Rust, you should
consider Memory Mapping (mmap) for the models.
   * Why: If you use mmap to load the ONNX or Candle weights, the OS handles the page
     cache. If the server crashes and restarts, the model is likely still in the OS page
cache (warm), making the "re-load" near-instant.
   * Consideration: Make sure the server and clients are clear on who "owns" the memory.
     For the IPC response, if you move to very large batches, Unix Sockets are fast, but
memfd_create (anonymous shared memory) is the "pro" way to pass large vectors between
processes without copying the bytes through the socket.

  3. Integrated GPU (AMD 680M) Nuances You mentioned no ROCm/CUDA, but on Linux, Vulkan
(via kompute or wgpu) is often a viable path for acceleration on integrated AMD chips.
   * Non-obvious consideration: Integrated GPUs share RAM with the CPU. When a model
     "loads" into GPU memory, it's just moving from one part of the 14GB to another, but
the Peak Memory Usage during the transfer can be double the model size. 
   * Improvement: The server should have a "Memory Guard" that checks /proc/meminfo
     before starting a load. If MemAvailable is too low, it should explicitly trigger a
malloc_trim or global GC before attempting to load the next model.

  4. Semantic Versioning in the Protocol The current JSON sketch doesn't have a version
field.
   * The Risk: You update the server to v2 (different embedding normalization), but
     Smriti is still on v1. Your search results will suddenly become "trash" because the
vectors aren't comparable.
   * Improvement: Every response should include a protocol_version and a
     model_fingerprint (maybe a hash of the weights). This ensures the tool knows
exactly what "version" of the embedding space it is writing to the database.

  5. The "Ready" Signal & Health Checks
   * Non-obvious consideration: Clients need a way to check "What is currently loaded?"
     without sending a full request. 
   * Improvement: Add a GET_STATUS or PING command. If Smriti sees that Qwen is loaded,
     it might choose to delay a background maintenance task to avoid forcing a model
switch, or it can warn the user that the "first search will take 5 seconds."

  6. Priority Queuing
   * Consideration: An interactive search from Smriti (User waiting) should always
     "pre-empt" a background indexing job from Vedakosha.
   * Improvement: The protocol should support a priority flag.
       * High: Interrupt current batch, switch model immediately.
       * Low: Wait for current model to go idle before switching.

  7. Logging & Observability
   * Consideration: When things feel "slow," you'll want to know if it's the inference
     time or the load/unload time.
   * Improvement: The response JSON should include a timing object:

       "timing": { "queue_wait_ms": 10, "model_load_ms": 4500, "inference_ms": 120 }
This will help you debug whether your "one-large-model" constraint is becoming a
bottleneck that justifies more RAM or better quantization.
