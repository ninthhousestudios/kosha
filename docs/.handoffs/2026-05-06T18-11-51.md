# kosha — handoff

Date: 2026-04-27 (evening)

## what's ready

### candle BF16 CPU matmul — PR submitted

PR: https://github.com/huggingface/candle/pull/3503
Fork: ninthhousestudios/candle, branch `bf16-cpu-matmul`, cloned at `~/soft/candle`
All candle-core tests pass. Covers gemm, MKL, and Accelerate backends.

### gemm-bf16 — local fork working

`~/soft/gemm` on branch `bf16` (checked out from PR #40 by gicrisf).
Tests pass, benchmarked: 1.1-2.8x faster than the upcast approach we used in the candle PR.
This is the proper root fix — converts during packing phase instead of whole-tensor upcast.

## pick up next

1. **Wire local gemm-bf16 into candle** — point candle's gemm dependency at `~/soft/gemm` (or a published fork), enable `bf16` feature, then simplify the candle matmul code to just add `DType::BF16` to the allow-list instead of the cast wrapper. This gives us the performance win from mixed-precision packing.

2. **Test full pipeline** — with BF16 matmul working, re-run fastembed Qwen3-VL-Embedding-2B in BF16 mode and verify embedding quality matches the PyTorch BF16 baseline (8x relevant/irrelevant ratio).

3. **Benchmark Qwen3-VL text quality vs BGE-M3** — needed to decide single-model vs dual-model architecture.

4. **RunPod batch embedding** — CLI workflow for GPU batch inference on image-heavy books.

## local repos

| Repo | Path | Branch | State |
|------|------|--------|-------|
| candle (fork) | ~/soft/candle | bf16-cpu-matmul | PR submitted |
| gemm (PR #40) | ~/soft/gemm | bf16 | tested, working |
| fastembed (fork) | ~/soft/fastembed-local | — | F16 dtype fixes, PR #248 filed |
| fastembed test | ~/soft/fastembed-qwen3-test | — | test harness |
| Qwen3-VL weights | ~/soft/Qwen3-VL-Embedding/ | — | PyTorch path works |

## blockers

- Both upstream PRs (candle #3503, gemm #40) waiting on maintainers — but local forks work
- Project dir rename vedakosha → kosha pending

## context

- GitHub account for contributions: ninthhousestudios
- Josh's machine: 14GB RAM, AMD Radeon 680M (integrated), CPU-only, Arch Linux
- Python ML: use uv, not system Python 3.14
- The qartez rewrite is separate, lower-priority
