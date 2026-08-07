# Matmul optimisation results

The implementation programme following [matmul-audit.md](matmul-audit.md) is complete. Raw baselines, profiles, candidate output and the full per-family table are under [`benchmarks/matmul`](../benchmarks/matmul/).

## End-to-end result

The pinned 11-second MOSS JFK fixture produced the same SHA-256 (`9a47c0f25721a7deddbdfb7efe651e2ee86f63a37776ca7319517d7bfed44928`) in every final CPU and GPU run.

| Path | Baseline median | Final median | Improvement | Final range |
|---|---:|---:|---:|---:|
| CPU/SIMD | 35.425s | 28.097s | **1.26x** | 27.212-28.869s |
| Automatic NVIDIA PTX | 12.990s | 12.892s | **1.01x** | 12.827-12.953s |

The GPU path remains **2.18x** faster than the final CPU path. Model loading is included in both measurements.

## Retained kernel gains

- Dense AVX2 NT: 1.43-1.52x at M=32/227 after 1x4 register tiling and K128/N64 blocks.
- Dense AVX2 NN: 1.53-1.92x across Whisper-like and prefill shapes using a 2x32 tile.
- BF16 output-row x4: 12.4-13.9x versus four scalar dots.
- GPTQ Q4 batch: 1.64-1.65x.
- MLX affine Q4 batch: 1.99-2.02x.
- NVFP4 CPU batch: 2.66-2.73x.
- FP8 CPU batch: up to 1.76x; LUT gather remains best for decode.
- GGUF Q2_K/Q3_K batch: 9.16-9.32x.
- F32 output-row x4: 1.20-1.49x on retained shape ranges.
- Whisper packed query batches: 1.26-1.39x, with 21MiB less scratch at sequence 1500.
- NVIDIA GPTQ Q4 batch: 3.90-4.11x over repeated GEMV.

## Rejected and deferred work

- ARM64 NEON 1x4 NT candidate failed real CIX P1 parity and was reverted.
- NVIDIA register-blocked/skinny SGEMM candidate failed live parity; the verified 16x16 kernel remains default.
- Three-pass online-softmax Whisper PTX passed parity but was 5-8x slower.
- IME2 persistent pooling was 10% slower than the single packed kernel.
- AICPU scheduling remains blocked by missing existing IME2/RVV symbols on arm64.
- Vulkan persistent GEMM remains deferred until its RMSNorm, partial RoPE and attention-score gates pass and a transfer-amortised API exists.
- NVFP4 GPU fusion remains deferred until its two-level scale semantics have an independently validated native kernel.

## Validation

The final focused matrix passes for SIMD runtime, GPTQ Q4, NVFP4, FP8, MLX, GGUF, tensor, BERT, Whisper, SpeechBrain, Hunyuan3D, Trellis2 and MOSS. Arm64 and riscv64 cross-builds pass for the shared SIMD and GGUF packages. Live RTX 3060 gates pass for Q4, FP8, the default SGEMM shape matrix and both Whisper attention candidates.

Known unrelated red gates remain documented rather than hidden: three Vulkan parity tests, the arm64 AICPU missing-symbol build, and the existing standalone Whisper arm64 `simdfft.Precompute*` build issue. The two Gemma4 K=V verifier failures recorded during this programme were subsequently fixed by deriving V from the original K projection before K-only normalisation; they are no longer broad-suite blockers.
