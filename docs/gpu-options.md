# GPU Compute Options

go-pherence currently has a production NVIDIA backend plus Vulkan backend scaffolding. NVIDIA/Vulkan use `purego` dlopen (no CGo):

## NVIDIA PTX (NVIDIA)

Primary GPU backend. 29 hand-written PTX kernels. Source strings are owned by `backends/nvidia/ptx`; runtime loading, launch helpers, `DevBuf`, and GPU-resident resources remain in the `backends/nvidia/runtime` package. Set `GO_PHERENCE_DISABLE_NVIDIA=1` to force CPU/non-NVIDIA behavior in tests or diagnostics:

| Category | Kernels | Notes |
|---|---|---|
| **Core GEMV** | sgemm_nn, gemv_q4sym, gemm_q4sym | GPTQ tiled + shared mem |
| **MLX** | mlx_gemv, mlx_gemm, mlx_correct | Transposed layout + bias |
| **Element-wise** | vec_add, vec_mul, vec_scale, vec_add_scaled, vec_silu | threshold-free |
| **Fused** | fused_silu_mul, rms_norm, gelu_tanh_mul | reduced launch count; standalone GELU/LayerNorm are not used by current model hot paths |
| **Norms** | rms_norm_no_scale, to_bf16_f32 | Gemma4 V-norm, BF16 trunc; RMSNorm is the model-owned norm primitive |
| **Attention** | rope_apply, rope_partial, row_softmax_debug, gqa_attention | precomputed cos/sin, scale param; standalone last-axis softmax is currently row-softmax only |
| **Utility** | lm_head_gemv, prefetch_l2, vec_scale | 2D grid, L2 warming |
| **BF16** | bf16_rms_norm, bf16_vec_add, bf16_silu_mul, bf16_gelu_tanh_mul | emulated (sm_80) |
| **BF16 native** | native_bf16_rms_norm/vec_add/gemv | ld.b16+cvt (sm_86+) |
| **NVFP4 fallback** | nvfp4_dequant_f32 | packed FP4 + F8_E4M3FN scales → F32 materialization |

Loaded as one mega module + optional native BF16 module.

### Memory Management

- **DevBuf**: device-agnostic buffers with lazy CPU↔GPU transfer; vector/norm/dense GEMV/LM-head fast paths preflight kernel operands before launching and fall back or no-op safely if upload/allocation fails; GPU-only scratch buffers avoid zero host uploads for temporary MoE workspaces
- **Quantized dispatch**: Q4/MLX upload paths validate dimensions, packed-weight sizes, scale layouts, and group indices before allocating GPU buffers; native MLX uploads can copy packed `uint32` weights directly when transposed GPTQ-compatible buffers are not needed; Q4 asymmetric and NVFP4 packed/native boundaries are documented in [nvidia-quant-boundaries.md](nvidia-quant-boundaries.md)
- **LM-head placement**: F32 LM-head is preferred for moderate heads when it fits; compact MLX LM-head is used for very large heads or low-VRAM-headroom cases
- **ExpertPool**: LRU cache for MoE expert weights with auto-sized VRAM budget; disabled and replacement cases return GPU resources for explicit release; Qwen-style MoE uses native-only MLX expert uploads, immediate GPU use on cache miss, and device-side expert-output accumulation
- **BudgetManager**: 4-tier memory tracking (resident/layer/stream/expert), now owned by `backends/placement`
- **MmapAdvisor**: `runtime/memory` page-level madvise tracking for eager loading and future weight streaming
- **Layer placement**: `backends/placement` auto-fit/manual policy (`--gpu-layers N`) with caller-supplied device memory availability

## Vulkan Compute (any GPU)

Portable backend for non-NVIDIA hardware. Vulkan code and shaders now live under `backends/vulkan`:

- **Targets**: Intel iGPU (UHD/Iris/Arc), AMD RDNA, ARM Mali, Qualcomm Adreno, MoltenVK
- **API**: 35 Vulkan functions, device auto-selection, compute queue + command pool
- **Shaders**: GLSL/SPIR-V coverage for vector add, RMSNorm, GEMV, SiLU, attention score, RMSNormNoScale, RoPEPartial, and GELU paths
- **BF16**: emulated via uint16 bitshift (no extensions needed)
- **Status**: init + buffer path, embedded SPIR-V, pipeline cache wiring, validating dispatch wrappers, and availability-gated CPU-vs-Vulkan parity tests are present for the covered primitives. See [backend-selection.md](backend-selection.md) for selection/fallback policy, [vulkan-dispatch-inventory.md](vulkan-dispatch-inventory.md) for the current shader/wrapper inventory, and [vulkan-validation-plan.md](vulkan-validation-plan.md) for the validation sequence.

### Fused-only primitive decisions

Current decoder hot paths do not justify separate NVIDIA LayerNorm or standalone GELU kernels:

- LLaMA/Qwen/Gemma decoder blocks use RMSNorm/RMSNormNoScale, already covered by NVIDIA runtime wrappers.
- GELU appears as the fused `gelu_tanh_mul` MLP activation path and Gemma4 per-layer-input gating path; no current production call site needs standalone GELU output materialization.
- Softmax GPU coverage is row-oriented for attention scores (`row_softmax_debug`) plus fused GQA attention. A generic last-axis softmax wrapper remains future work only if a tensor/model path needs it beyond row-softmax.

## NVFP4 / FP4 Track

NVFP4 is an experimental/internal NVIDIA quantization path. Public NVIDIA
ModelOpt and community checkpoints exist for Qwen3 and Gemma4, but go-pherence
still rejects them during public model loading. Synthetic CPU-vs-NVIDIA dequant
parity now passes; real checkpoint logits/tokens remain the enablement gate:

- loader detection for ModelOpt / compressed-tensors metadata is in place, including mixed `config_groups`, `format`, `weights.format`, and 4-bit float `weights.type` variants
- Qwen3 dense, Qwen3 MoE, and Gemma4 tensor naming/layout metadata is documented
- `backends/simd/quant/nvfp4` owns correctness-first FP4/F8 decode, dequant, GEMV, and
  synthetic-logit tests
- `backends/nvidia/runtime` has `GPUNVFP4Weight`, raw byte upload, NVIDIA dequant-to-F32 fallback,
  native tensor-core capability gating, dense GEMV fallback via F32 materialization, and a packed GEMV/GEMM `NVFP4KernelSpec` contract with u32/overflow guards
- packed/native GEMV/GEMM and MoE expert cache integration remain future work

See [nvfp4.md](nvfp4.md) for current model-weight findings and roadmap.

## CPU SIMD Assembly

AVX2+FMA (amd64) and NEON (arm64):

- Runtime-gated AVX2/FMA and NEON wrappers with scalar fallback
- Covered hot paths include vector add/mul/scale, dot/Saxpy, RMSNorm variants, BF16 widen/narrow, SGEMM wrappers, the Ideogram CPU FP8 E4M3 LUT gather-dot GEMV kernel on amd64 AVX2/FMA, NVIDIA direct FP8 E4M3 GEMV for opt-in Ideogram linear streaming/lazy residency, NVIDIA fused Ideogram CFG+FlowMatch scheduler vector update, NVIDIA Ideogram non-affine row LayerNorm, low-level F32 RMSNorm Buffer wrappers for Ideogram normalization wiring, NVIDIA Ideogram adaLN scale/gate plus gated-residual vector kernels, NVIDIA Ideogram full-tensor MRoPE rotation, NVIDIA Ideogram full non-causal DiT attention score/softmax/value kernels, and NVIDIA Ideogram MLP/final-vector SiLU/Mul/SiLU*Mul wrappers
- Remaining CPU SIMD gaps include fused GELU, RoPEPartial, and MLX/GPTQ Q4 GEMV kernels

## Backend Selection

```
if NVIDIA GPU available and enabled:
    → NVIDIA PTX (fastest, 29 kernels)
elif an explicit Vulkan wrapper is used and a non-software Vulkan device is available:
    → backends/vulkan SPIR-V for covered primitives
else:
    → CPU SIMD (AVX2 or NEON assembly)
    → Go scalar (universal fallback)
```

The current production LLM model path chooses NVIDIA when requested/available, otherwise CPU SIMD/scalar. Vulkan wrappers are usable for the covered primitive tests and explicit dispatch calls, but model-level Vulkan placement is still intentionally opt-in/experimental. Ideogram 4 is a separate image pipeline: its default path remains CPU/SIMD fp8/DiT/VAE, with opt-in correctness-oriented FP8 linear streaming/lazy residency via `GO_PHERENCE_IDEOGRAM4_GPU_FP8=1` and `GO_PHERENCE_IDEOGRAM4_GPU_FP8_CACHE=1`, fused CFG/scheduler update via `GO_PHERENCE_IDEOGRAM4_GPU_CFG=1`, normalization/adaLN kernels via `GO_PHERENCE_IDEOGRAM4_GPU_NORM=1`, MRoPE rotation via `GO_PHERENCE_IDEOGRAM4_GPU_MROPE=1`, full DiT attention via `GO_PHERENCE_IDEOGRAM4_GPU_ATTN=1`, and MLP/final vector ops via `GO_PHERENCE_IDEOGRAM4_GPU_MLP=1`; no full CUDA/NVIDIA Ideogram graph has been wired yet. See [backend-selection.md](backend-selection.md) for detailed gates, fallback rules, and wrapper coverage.

### Debug and diagnostics gates

Library/backend progress diagnostics are quiet by default. Opt in when debugging backend discovery or placement:

- `GO_PHERENCE_GPU_DEBUG=1` — NVIDIA backend init, module, stream, native-BF16, and experimental direct-NVIDIA ioctl diagnostics.
- `GO_PHERENCE_VULKAN_DEBUG=1` — Vulkan discovery, CPU-device rejection, device creation, and pending-SPIR-V diagnostics.
- `GO_PHERENCE_LOAD_DEBUG=1` — model loader, quantization detection, GPU placement, LM-head, expert-pool, and VRAM budget diagnostics.
- `GO_PHERENCE_PREFILL_DEBUG=1` — batched-prefill progress diagnostics.
- `GO_PHERENCE_VULKAN_ALLOW_CPU=1` — allows CPU/software Vulkan implementations such as llvmpipe for shader/backend debugging; by default these are rejected so `--gpu` does not silently select a CPU device.


## DevBuf/dispatch guard status

The `backends/nvidia/runtime` package is hardened at backend API boundaries:

- `DevBuf` receiver helpers are nil-safe; `ToGPU`/`GPUPtr` propagate upload failures, failed downloads keep GPU contents authoritative, slice views use overflow-safe byte math, and copy helpers fall back instead of marking stale state authoritative.
- NVIDIA allocation/upload/download/copy helpers share checked byte-size arithmetic, reject non-empty copies to zero-sized buffers, and validate D2D copies before driver calls.
- Stream/graph helpers validate nil graph executables, nil kernel arguments, and invalid launch dimensions before NVIDIA calls.
- Q4/MLX quantized weight upload/dispatch validates packed-weight and scale product arithmetic, buffer byte sizes, group consistency/indices, batched dimensions, and download errors in CPU fallback.
- Expert-pool helpers reject nil pools and invalid expert IDs without leaking caller-owned GPU resources.
- Experimental direct-NVIDIA ioctl/memory/query/GPFIFO helpers validate nil receivers, size arithmetic, fd/argument state, class-list sizes, and release partially allocated resources on setup failure.
- Dense SGEMM/LM-head dispatch validates dimensions, buffer byte sizes, and product overflow before kernel launch.
- NVIDIA JIT helpers validate kernel specs and launch buffers with the same checked byte-size helper before PTX generation or dispatch.
- BF16 NVIDIA wrappers validate nil/undersized buffers and length overflow before emulated/native dispatch; parity expectations for no-scale RMSNorm and BF16 LM-head are tracked in [bf16-parity.md](bf16-parity.md).
- RoPE, partial RoPE, softmax-row, and GQA attention wrappers validate dimensions, sequence windows, tensor lengths, and product overflow before launch.

These guards are part of the current backend baseline and should stay with the NVIDIA runtime package as it continues to be refined.
