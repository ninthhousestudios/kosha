# candle BF16 CPU matmul — PR sketch

Status: sketch
Date: 2026-04-27
Target repo: https://github.com/huggingface/candle
Related issue: #920 (open since 2023)

## the problem

candle's CPU backend does not support BF16 matrix multiplication. The allow-list in `candle-core/src/cpu_backend/mod.rs` (around line 1367) only permits F16, F32, F64:

```rust
match T::DTYPE {
    DType::F16 | DType::F32 | DType::F64 => {}
    _ => Err(Error::UnsupportedDTypeForOp(T::DTYPE, "matmul").bt())?,
}
```

The underlying `gemm` crate (sarah-ek/gemm v0.19) has `gemm-f16`, `gemm-f32`, `gemm-f64` but no `gemm-bf16`. So even adding BF16 to the match arm would panic inside gemm at runtime.

This blocks running any BF16 model on CPU — including Qwen3-VL-Embedding-2B via fastembed's candle backend. F16 works mechanically but degrades embedding quality (2x relevant/irrelevant ratio vs 8x with BF16, tested empirically).

## the fix: cast-to-F32 wrapper

Intercept BF16 matmul, cast both operands to F32, call the existing F32 gemm path, cast the result back to BF16. This is the same pattern `gemm-f16` uses internally (accumulate in F32), just without the SIMD-optimized packing loops.

### where to change

`candle-core/src/cpu_backend/mod.rs`, in the `MatMul` implementation. The change needs to happen *before* entering the generic `f<T: WithDType>` because inside that generic, `T` is `bf16` and we need to work with `f32`. Two approaches:

**Approach 1: Special-case in `Map2::map` dispatch (cleanest)**

In `src/cpu_backend/utils.rs`, the `Map2::map` method matches on `(CpuStorage::BF16(v1), CpuStorage::BF16(v2))`. Instead of calling `self.f(v1, l1, v2, l2)` which enters the generic path, intercept here:

```rust
(C::BF16(lhs), C::BF16(rhs)) => {
    // Convert BF16 slices to F32
    let lhs_f32: Vec<f32> = lhs.iter().map(|v| v.to_f32()).collect();
    let rhs_f32: Vec<f32> = rhs.iter().map(|v| v.to_f32()).collect();
    // Run the F32 matmul
    let result_f32 = self.f(&lhs_f32, lhs_l, &rhs_f32, rhs_l)?;
    // Convert back to BF16
    let result_bf16: Vec<bf16> = result_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    Ok(C::BF16(result_bf16))
}
```

Problem: `Map2::map` is generic across all ops, not just matmul. This intercept would apply to every binary op, which is wrong.

**Approach 2: Special-case in `MatMul::f` (more surgical)**

Add a BF16 branch inside `MatMul::f()` that bypasses the generic gemm call:

```rust
match T::DTYPE {
    DType::F16 | DType::F32 | DType::F64 => {
        // existing gemm path
    }
    DType::BF16 => {
        // Cast lhs/rhs slices to f32 via unsafe reinterpret + convert
        // Call gemm with f32 slices
        // Cast result back to bf16
        // Write into dst
    }
    _ => Err(Error::UnsupportedDTypeForOp(T::DTYPE, "matmul").bt())?,
}
```

The tricky part: inside `f<T>`, `T` is `bf16` but we need `f32` slices. We know `T::DTYPE == DType::BF16`, so we can use `half::bf16` directly via unsafe pointer casts, or factor the conversion into a helper that works at the byte level.

**Recommended: Approach 2** — it's surgical (only affects matmul), doesn't change the generic dispatch, and the unsafe is well-contained.

### the actual conversion

```rust
fn bf16_slice_to_f32(src: &[bf16]) -> Vec<f32> {
    src.iter().map(|v| v.to_f32()).collect()
}

fn f32_slice_to_bf16(src: &[f32]) -> Vec<bf16> {
    src.iter().map(|v| bf16::from_f32(*v)).collect()
}
```

For the matmul itself, the existing code computes into a `dst: &mut [T]` output buffer. The BF16 path would:
1. Allocate temporary `Vec<f32>` buffers for lhs, rhs, and dst
2. Convert lhs and rhs from bf16 to f32
3. Call `gemm::gemm()` with the f32 buffers
4. Convert the f32 dst back to bf16 and write into the original bf16 dst

### memory cost

The temporary f32 buffers hold the matmul operands (not the whole model). For a typical transformer layer with hidden_size=1536 (Qwen3-VL-2B):
- Largest matmul: ~1536 × 1536 × 4 bytes = ~9 MB per operand
- Three temporary buffers (lhs, rhs, dst): ~27 MB
- This is negligible compared to the model's ~4 GB

### performance cost

- bf16→f32 conversion: O(n) where n = elements in the operand
- f32 gemm: O(n³) for the matmul
- f32→bf16 conversion: O(n) for the output

For any matrix larger than ~32×32, the O(n³) gemm dominates. The conversion overhead is negligible.

### what NOT to do

- Don't add bf16 to the existing match arm without gemm support — it will panic at runtime
- Don't try to add native BF16 SIMD — that's a separate effort in the gemm crate (AVX-512 BF16 instructions, etc.)
- Don't change the MKL or Accelerate paths — those have their own BLAS calls and don't need this

## testing

candle has matmul tests in `candle-core/tests/matmul_tests.rs`. Add a test:

```rust
#[test]
fn matmul_bf16() -> Result<()> {
    let a = Tensor::randn(0f32, 1., (4, 3), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let b = Tensor::randn(0f32, 1., (3, 5), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dtype(), DType::BF16);
    assert_eq!(c.dims(), &[4, 5]);
    // Verify against F32 reference
    let a_f32 = a.to_dtype(DType::F32)?;
    let b_f32 = b.to_dtype(DType::F32)?;
    let c_ref = a_f32.matmul(&b_f32)?;
    let c_f32 = c.to_dtype(DType::F32)?;
    let diff = (c_f32 - c_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 0.1, "BF16 matmul result differs from F32 reference by {diff}");
    Ok(())
}
```

## scope and effort

- ~30-50 lines of new code in `cpu_backend/mod.rs`
- ~20 lines of test
- No changes to the gemm crate
- No changes to CUDA/Metal backends
- No new dependencies (half crate is already a dependency)
- Weekend-sized PR

## also fix: other BF16 binary ops

The same pattern (BF16 not in the allow-list) likely affects other ops beyond matmul. A follow-up PR could audit all ops in `cpu_backend` that have dtype allow-lists and add BF16 cast-to-F32 paths where missing. But matmul is the blocker — the other ops can be addressed incrementally.
