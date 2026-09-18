# Backend stack

The CPU path is the correctness baseline. Other backends become defaults only for the model surfaces where they have numerical and end-to-end parity; availability alone is not enough.

## CPU and SIMD

`backends/simd/runtime` is the public CPU facade. It validates shapes and backing storage, dispatches to AVX2, NEON or RVV when available, and otherwise calls the scalar implementation in `backends/simd/kernels` or the relevant quantised package.

The hot paths cover dense NN/NT matrix multiplication, output-row x4 GEMV, norms, vector operations, attention helpers and BF16 conversion/dot products. Quantised CPU ownership is split by format:

| Format | Owner |
|---|---|
| MLX affine quantisation | `backends/mlx` |
| GPTQ Q4 | `backends/simd/quant/q4` |
| NVFP4 | `backends/simd/quant/nvfp4` |
| FP8 E4M3 | `backends/simd/quant/fp8` |
| GGUF block formats | `loader/gguf` with backend helpers |

Decode and prefill use different kernels. Decode is normally bandwidth-bound at M=1; prompt and verifier batches use multi-output register tiles and cache blocking. [SIMD matmul policy](simd-matmul.md) explains the dispatch, and [matmul optimisation results](matmul-optimisation-results.md) records the measured ranges.

## NVIDIA PTX

`backends/nvidia/runtime` loads the CUDA driver through purego, owns device buffers, streams, modules and model residency, and dispatches kernels embedded under `backends/nvidia/ptx` and the Whisper-oriented `backends/cuda/ptx`. No CUDA toolkit or CGo runtime is required.

Coverage includes dense and quantised GEMV/GEMM, norms, activations, RoPE, attention, LM heads, BF16, FP8, NVFP4 support surfaces and expert caches. Model code decides which of those pieces form a verified resident graph; this is why MOSS can select its complete NVIDIA path automatically while some standalone Whisper and image-model surfaces remain opt-in.

Diagnostics are off by default. Use `GO_PHERENCE_LOAD_DEBUG=1` for weight placement and `GO_PHERENCE_GPU_DEBUG=1` for driver/kernel dispatch.

[NVIDIA quantisation boundaries](nvidia-quant-boundaries.md), [NVFP4](nvfp4.md), [BF16 parity](bf16-parity.md) and [GPU options](gpu-options.md) contain the format and command details.

## Vulkan

`backends/vulkan` owns loader/device lifetime, bounded memory arenas, inspected embedded SPIR-V, checked operations and multi-stage plans. Its explicit F32 speech surfaces include a resident Whisper encoder (Conv1D, add, LayerNorm, linear, erf-GELU and non-causal attention) and Community-1 resident LSTM/ResNet owners. The separate legacy GQA attention-score wrapper now dispatches the complete head×time grid and rejects invalid head ratios; it is not the Whisper encoder's fused attention implementation. These paths copy fixed-shape weights once, keep intermediate tensors resident and do not silently fall back to CPU; model policy remains responsible for selecting a qualified graph. Legacy vector/RMSNorm/RoPE/GEMV wrappers remain separate from those speech graphs.

Vulkan is not a general automatic model backend: speech paths are opt-in, quantised/F16 speech graphs and device recreation remain open, and native/trained quality and placement evidence is model-specific. Software devices are rejected unless `GO_PHERENCE_VULKAN_ALLOW_CPU=1` is set; use that switch for debugging, never acceleration measurements.

See [Vulkan inventory](vulkan-dispatch-inventory.md) and [Vulkan validation](vulkan-validation-plan.md).

## SpacemiT, CIX and RISC-V

`backends/spacemit` combines three kinds of execution:

* RVV CPU kernels for dense, FP16 and quantised matrix multiplication.
* IME2 `vmadot` kernels with pre-packed INT8 tiles.
* AICPU/A100/X100 experiments for pooled and model-specific packed work.

The aligned RVV and IME2 cores are already multi-output kernels. Recent work added safe M/N tails around the RVV packed path; persistent IME2 pooling was measured on the CIX P1 and rejected because one packed GEMM was 10% faster. AICPU scheduling work is currently blocked by missing IME2/RVV symbols in the arm64 build.

[SpacemiT IME2](spacemit-ime2.md) is the canonical hardware guide; [Whisper on RISC-V](whisper-riscv-optimization.md) and [Ideogram on SpacemiT](ideogram4-spacemit.md) contain model results.

## Placement and fallback

`backends/placement` estimates layer residency from caller-supplied memory budgets. Individual models add constraints for KV state, LM heads, experts and temporary buffers. Failed optional acceleration returns to CPU unless the command explicitly requires the device path.

Use [Backend selection](backend-selection.md) for selection order, [Tuning](tuning.md) for command switches and [Weight budgets](weight-budget.md) for memory policy. Source ownership lives in [Backend layout](backend-layout.md); primitive-level readiness lives in [Kernel coverage](kernel-coverage.md).
