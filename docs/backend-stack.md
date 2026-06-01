# Backend stack

This page summarizes backend execution paths and package ownership.

## NVIDIA GPU backend

Hand-written PTX kernels are compiled by the driver at runtime via `purego`/`dlopen`. Runtime dispatch/resource ownership lives in `backends/nvidia/runtime`; embedded PTX source assets live in `backends/nvidia/ptx`, with BF16/Q4/MLX/NVFP4 grouped by quantization.

Current coverage includes:

- Quantized GEMV for GPTQ and MLX affine 4-bit weights.
- Batched GEMM/prefill scaffolding for supported paths.
- LM-head placement: F32 when moderate and resident, compact MLX for very large heads or tight VRAM.
- RMSNorm, RoPE, GQA attention, SiLU, vector add/mul/scale/add-scaled.
- BF16 kernels with native `cvt.f32.bf16` on Ampere+ and emulation on older paths.
- MLX bias correction and packed-weight upload helpers.
- NVFP4/FP4 experimental upload and correctness-first fallback kernels.
- GPU-resident expert cache for MoE paths.
- Hybrid CPU/GPU forward via `--gpu-layers N`.
- Configurable GPU KV horizon via `--gpu-kv-max-seq N` / `GO_PHERENCE_GPU_KV_MAX_SEQ` for prompt/MTP smokes.

Diagnostics are quiet by default. Use `GO_PHERENCE_LOAD_DEBUG=1` or `GO_PHERENCE_GPU_DEBUG=1` when inspecting placement and CUDA behavior.

## Vulkan backend

`backends/vulkan` owns the Vulkan loader, device/buffer helpers, dispatch scaffolding, and embedded SPIR-V assets.

Current status:

- 35 Vulkan API functions via `purego`; no SDK is required at runtime.
- GLSL/SPIR-V assets for vector add, RMSNorm, GEMV, SiLU, attention score, RMSNormNoScale, RoPEPartial, and GELU paths.
- Device auto-selection rejects software/CPU devices by default.
- Set `GO_PHERENCE_VULKAN_ALLOW_CPU=1` for software-device debugging.
- Full model dispatch wiring remains pending; wrapper/availability tests are in place.

See also [gpu-options.md](gpu-options.md), [vulkan-dispatch-inventory.md](vulkan-dispatch-inventory.md), and [vulkan-validation-plan.md](vulkan-validation-plan.md).

## CPU SIMD backend

`backends/simd/runtime` is the public facade. Optimized kernels and scalar references live under `backends/simd/*`; non-SIMD packages should import the runtime facade, not private kernels.

| Operation | AVX2 | NEON | Notes |
|---|---|---|---|
| **Sdot** | 16-wide FMA | 8-wide VFMLA | Dot product. |
| **RMSNorm** | fused sum-sq + scale | 4-wide | Hot norm path. |
| **VecAdd** | 16-wide VADDPS | 8-wide FADD | Vector add. |
| **ToBF16** | 16-wide VANDPS | 8-wide AND | Truncate/narrow helper. |
| **BF16 Dot** | widen+FMA+narrow | USHLL+VFMLA | BF16 hot path. |
| **BF16 RMSNorm** | fused BF16 | USHLL+XTN | BF16 norm. |
| **BF16 Widen** | VPMOVZXWD+VPSLLD | USHLL+SHL | BF16 to F32. |
| **SiLU×Mul / GELU×Mul** | scalar/reference today | scalar/reference today | Optimized polynomial kernels are pending. |

Public SIMD entrypoints are runtime-gated with scalar fallback. Remaining CPU gaps include true AVX2/NEON fused GELU/SiLU kernels, RoPEPartial kernels, and MLX/GPTQ Q4 assembly kernels.

## Native BF16

End-to-end BF16 support spans the loader, SIMD facade, NVIDIA backend, Vulkan scaffolding, and model helpers:

- Safetensors `GetBF16()` returns `[]uint16` without F32 conversion.
- AVX2/NEON helpers cover BF16 dot/norm/add/widen/narrow.
- NVIDIA uses native `ld.global.b16` / `cvt.f32.bf16` where available.
- Vulkan emulates BF16 via uint16 bitshift.
- Model paths retain F32-compatible fallbacks where required.

## Package ownership overview

- `loader/` — config, tokenizer, safetensors, GGUF inspection/tokenizer helpers, and shared weight-source opening.
- `backends/placement/` — backend-neutral memory budget and layer placement policy.
- `backends/simd/` — AVX2/FMA and NEON dispatch/kernels plus checked scalar fallbacks.
- `backends/nvidia/ptx/` — PTX source assets grouped by quantization family.
- `backends/nvidia/runtime/` — NVIDIA runtime, DevBuf, kernels, GPU weights, expert cache, and diagnostics.
- `backends/vulkan/` — Vulkan loader/device/buffer/shader dispatch scaffolding.
- `models/bert/` — GTE/BERT encoder path.
- `runtime/kv/` — TurboQuant state, compressed KV cache, shared KV byte estimator, and staging/rollback primitives.
- `runtime/memory/` — mmap residency advice and tracked range accounting.
- `runtime/quant/` — compatibility wrappers for backend-owned quantization implementations.
- `model/`, `model/qwen/`, `model/gemma4/`, `model/llama/` — shared LLaMA-family decoder plus focused architecture packages/diagnostics, including native GGUF REAP/TurboQuant planning and smoke validation surfaces.

See [backend-layout.md](backend-layout.md), [kernel-coverage.md](kernel-coverage.md), and [backend-parity-matrix.md](backend-parity-matrix.md) for detailed status.
