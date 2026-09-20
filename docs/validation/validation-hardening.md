# Validation and hardening status

[Post-audit host remediation](repository-safety-host-remediation-20260920.md) fixes legacy cache/sidecar admission and concurrent ownership, speakercheck's scoring/temp/time boundaries, prepared-prompt accounting and a speaker merge-label defect. Zero legacy cache budget now disables storage; native issues and remaining host admission work stay open.

The [package audit closeout](repository-safety-audit-closeout-20260920.md) records selected source-boundary review across164/164 packages. [Open findings](repository-safety-audit-open-findings-20260920.md) remain tracked in #16 and platform/backend issues #17--#21; this is not exhaustive correctness or device clearance.

The [follow-on audit](repository-safety-followon-20260919.md) adds mapped-copy/Close coordination, HTTP bounds, worker joins, finite numeric comparison and graph/CPU-fallback regressions. Raw mmap views still require owner lifetime, and K3 teardown is compile-only off-device.

The [repository audit](repository-safety-audit-20260919.md) and [coverage matrix](repository-safety-audit-coverage-20260919.md) distinguish implemented guards from inspected source and unavailable hardware validation. These checks do not establish concurrent model/Close safety, capture-owner isolation or exhaustive malformed-input coverage.

The [fifth pass](repository-safety-fifth-pass-20260920.md) adds aggregate helper/stream limits, fake-tested ioctl descriptor ownership, scalar RoPE and resampling bounds, and fail-fast GLiNER loading. The [sixth pass](repository-safety-sixth-pass-20260920.md) fixes the reproduced Qwen3-TTS planning/config issues and adds packed-Q4, BF16 and compressed-cache regressions. Native TCM/C-shim and GGML ownership/extent findings remain open, not validated away by host tests.

The [seventh pass](repository-safety-seventh-pass-20260920.md) coordinates mmap advice with unmap, fixes overlap/eviction bookkeeping, preserves FP16 NaNs through scalar/SIMD GELU dispatch, and checks LLaMA configs, sparse rows and Qwen key/planner loops. Retained raw views, native library lifecycle, frozen tokenizer semantics and older prompt-sidecar budgeting remain distinct gaps.

This page summarizes recent malformed-input and boundary-hardening work. Phase-level commands live in [validation-gates.md](validation-gates.md); detailed coverage tables live in [malformed-input-coverage.md](malformed-input-coverage.md), [kernel-coverage.md](../architecture/kernel-coverage.md), and [final-coverage-acceptance.md](../history/final-coverage-acceptance.md).

## Runtime and tensor layers

- `tensor/` validates shapes, reductions, broadcasting, unsafe float32 views, realization internals, rewrite/fusion graphs, pooled allocations, NN helpers, convenience ops, embeddings, matmul/linear helpers, and module wrappers.
- `runtime/kv` guards cache dimensions/layouts, compressed-cache accessors, stored/scratch/total memory accounting, aggregate compressed-cache stats, staging rollback arithmetic, TurboQuant sizing, packed-byte calculations, protected-layer helper inputs, SIMD rotation capability reporting, scratch-aware quant/dequant helpers, and nil/malformed cache receivers.
- `runtime/memory` guards mmap range overflow, malformed tracked ranges, nil advisor receivers, and saturating accounting.

## Quantization boundaries

- Backend-owned quantization packages validate MLX/GPTQ/Q4 tensor layouts, shape/expected-size/dequant output arithmetic, NVFP4 unpack/dequant bounds, and malformed in-memory weights.
- `runtime/quant` remains a legacy compatibility re-export layer only; backend/model/tensor code should import owning backend packages directly.

## Loader boundaries

- `loader/gguf` inspection now reports and validates native GGUF REAP/TurboQuant readiness inputs: architecture/name, tensor and quant inventory, REAP ratio/source, hidden/head/vocab/tokenizer/BOS/EOS/context/KV shape, MoE expert counts, and full-attention/cache/protected-layer planning for QwenNext-style REAP checkpoints.
- `loader/safetensors` validates dtype byte sizes against shapes/offsets at open time.
- File and sharded helpers are nil-safe.
- Tensor names are sorted deterministically.
- Partial sharded opens clean up already-open shards.
- Sharded eager-load totals are checked.
- Tokenizer byte maps are initialized with `sync.Once`.
- Malformed tokenizer BPE merges are rejected.

The latest shared-loader checks also reject malformed WAV format/rate/alignment,
truncated data and bad chunk padding; prevent sharded path escapes under an
immutable-filesystem contract; use an atomic safetensors prefetch sink; and refuse
metadata fallback from a broken index to another checkpoint. Explicit index paths
are supported. GGUF semantic header rejection now closes the descriptor before
return, with a failing-before/passing-after regression. Raw mmap slices remain
borrowed and must not outlive or race Close.

## NVIDIA runtime

`backends/nvidia/runtime` preflights:

- dimensions and byte-size arithmetic,
- upload/download state,
- device pointers,
- stream launches and graph executables,
- copy wrappers,
- Q4/MLX/NVFP4 weight layouts,
- expert IDs and expert-pool uploads,
- NVIDIA ioctl/memory/query setup,
- dense SGEMM/LM-head buffers,
- JIT/NVFP4 kernel specs,
- BF16 buffers before dispatch.

Failed `DevBuf` transfers preserve authoritative state or fall back safely. NVIDIA progress diagnostics are quiet unless `GO_PHERENCE_GPU_DEBUG` is set.

Context-dependent driver calls now retain the OS-thread pin and driver lock
through launch/copy/module/stream/event operations. Foreign-call argument owners
remain live; JIT cache keys include constants and topology, launch buffer counts
must match the ABI, and shutdown invalidates lazy functions. KV copies join the
capture stream. These host/fake-driver checks are not GPU memory-safety proof;
global capture/scratch ownership and quiescent-only Shutdown remain open.

## Vulkan boundaries

Malformed geometry is rejected before optional pipeline initialisation. Shader
index products fit uint32 and dispatch ceil-division cannot wrap. Availability
errors on valid buffers are distinct from invalid-input errors; no failed GPU
test is silently counted as device validation.

## SIMD runtime

`backends/simd/runtime` scalar fallbacks bound all input/output slices. Additional guards include:

- BF16 GEMV shape-product overflow checks,
- precise `math.Sqrt` scalar RMSNorm,
- empty vector/BF16 calls avoiding assembly stubs,
- per-call GEBP scratch,
- SGEMM/GEBP/gather preflights for dimensions, pointers, strides, CPU capability gates, checked byte offsets, and overflow,
- FP8 E4M3 linear validation for weight/scale/bias shape consistency, with amd64 AVX2/FMA dot dispatch guarded by CPU feature checks, scalar fallback elsewhere, an opt-in NVIDIA `GPUFP8E4M3Linear` upload/GEMV/cache boundary that validates byte capacities, scale/bias buffers, CUDA u32 dimensions, explicit `ReleaseGPU` cleanup behavior, and CPU fallback behavior, a fused Ideogram CFG+FlowMatch NVIDIA vector boundary with shape/capacity/u32 checks and CPU fallback, an Ideogram non-affine LayerNorm NVIDIA boundary with row/column/capacity/u32 checks and CPU fallback, a low-level F32 RMSNorm NVIDIA boundary with input/weight/output capacity checks, Ideogram adaLN/gated-residual NVIDIA boundaries with emb/count/capacity/u32 checks, an Ideogram MRoPE NVIDIA boundary with token/head/head-dim/table/capacity/u32 checks, Ideogram full-attention score/probability/value NVIDIA boundaries with token/head/head-dim/product/capacity/u32 checks, and Ideogram MLP/final-vector SiLU/Mul/SiLU*Mul boundaries with length/capacity/u32 checks.

## Model helpers

Transitional model helpers validate:

- MTP token/KV keep counts,
- model-aware verifier plan/logit/activation dimensions,
- shared-KV verifier sources,
- MTP acceptance consistency before KV commit,
- alias-safe drafter projection sizing,
- q-only drafter external-KV/layer dimensions,
- bounded multi-draft counts,
- speculative stats overflow/rollback paths,
- zero-count state copy semantics,
- CPU decode final norm/LM-head dimensions,
- CPU generation allocation setup,
- GGUF generation KV allocation and validation paths, including expected greedy token/decoded text, static TurboQuant full/estimated/saved/scratch/total byte plans, runtime KV+scratch byte plans, native SIMD rotation readiness assertions, benchmark aggregate KV counters, and synthetic compressed-cache stored/scratch/total smoke accounting via `ggufsmoke`/Make targets.
- MoE edge cases,
- embedding/LM-head/per-layer input backing data,
- chunked LM-head and batched-prefill dimensions,
- CPU forward-layer entrypoints,
- model-specific KV width overflow,
- low-level GEMV/GQA product arithmetic.

Loader, prefill, and GPU placement diagnostics are quiet unless `GO_PHERENCE_LOAD_DEBUG` or `GO_PHERENCE_PREFILL_DEBUG` is set.
