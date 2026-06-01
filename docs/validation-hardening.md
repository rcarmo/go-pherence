# Validation and hardening status

This page summarizes recent malformed-input and boundary-hardening work. Phase-level commands live in [validation-gates.md](validation-gates.md); detailed coverage tables live in [malformed-input-coverage.md](malformed-input-coverage.md), [kernel-coverage.md](kernel-coverage.md), and [final-coverage-acceptance.md](final-coverage-acceptance.md).

## Runtime and tensor layers

- `tensor/` validates shapes, reductions, broadcasting, unsafe float32 views, realization internals, rewrite/fusion graphs, pooled allocations, NN helpers, convenience ops, embeddings, matmul/linear helpers, and module wrappers.
- `runtime/kv` guards cache dimensions/layouts, compressed-cache accessors, memory accounting, staging rollback arithmetic, TurboQuant sizing, packed-byte calculations, protected-layer helper inputs, and nil/malformed cache receivers.
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

## SIMD runtime

`backends/simd/runtime` scalar fallbacks bound all input/output slices. Additional guards include:

- BF16 GEMV shape-product overflow checks,
- precise `math.Sqrt` scalar RMSNorm,
- empty vector/BF16 calls avoiding assembly stubs,
- per-call GEBP scratch,
- SGEMM/GEBP/gather preflights for dimensions, pointers, strides, CPU capability gates, checked byte offsets, and overflow.

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
- GGUF generation KV allocation and validation paths, including expected greedy token/decoded text, runtime KV byte plan, benchmark KV counters, and synthetic compressed-cache smoke accounting via `ggufsmoke`/Make targets.
- MoE edge cases,
- embedding/LM-head/per-layer input backing data,
- chunked LM-head and batched-prefill dimensions,
- CPU forward-layer entrypoints,
- model-specific KV width overflow,
- low-level GEMV/GQA product arithmetic.

Loader, prefill, and GPU placement diagnostics are quiet unless `GO_PHERENCE_LOAD_DEBUG` or `GO_PHERENCE_PREFILL_DEBUG` is set.
