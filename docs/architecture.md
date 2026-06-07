# Architecture

go-pherence is a multi-backend inference engine that runs MLX, GPTQ, and BF16 model weights on any hardware.

## Design Goals

1. **Run MLX weights everywhere** — Apple's MLX ecosystem has the best quantized models, but only runs on Apple Silicon. go-pherence makes them portable.
2. **Pure Go, zero CGo** — single static binary, GPU activates at runtime via `purego` dlopen.
3. **Tiered acceleration** — production NVIDIA PTX path, Vulkan SPIR-V scaffolding, SIMD assembly, and Go scalar fallback.
4. **Native BF16 scaffolding** — half-bandwidth helpers for BF16-trained models, with F32-compatible paths still used where required.

## Backend Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Model Forward Pass                    │
│  (llama, qwen2, qwen3, gemma3, gemma4)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ NVIDIA PTX │  │  Vulkan  │  │   SIMD   │  ┌────────┐ │
│  │ 21 kernels│ │ SPIR-V   │  │AVX2+NEON │  │Go scalar│ │
│  │ sm_80+   │  │ any GPU  │  │asm       │  │fallback │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│       │              │             │             │       │
│  ┌────┴────┐   ┌─────┴────┐  ┌────┴────┐   ┌───┴───┐  │
│  │NVIDIA   │   │Intel iGPU│  │x86_64   │   │any    │  │
│  │RTX/GTX  │   │AMD RDNA  │  │ARM64    │   │GOARCH │  │
│  │Jetson   │   │Mali/Adr. │  │         │   │       │  │
│  └─────────┘   └──────────┘  └─────────┘   └───────┘  │
└─────────────────────────────────────────────────────────┘
```

## Source Layout

Phase 6.5 has moved the repository toward explicit ownership boundaries. Remaining large splits are documented follow-ups rather than Phase 6.5 blockers:

| Area | Current package | Notes |
|---|---|---|
| CLI front-ends | `cmd/llm/llmgen`, `cmd/llm/llmchat`, `cmd/llm/llmserver`, `cmd/llm/specbench`, `cmd/llm/speccheck`, `cmd/qwen/qwenmtpmeta`, `cmd/qwen/qwenmtpsynth` | Flags and user/server I/O only; `specbench` emits normal-vs-speculative CSV benchmark rows, `speccheck` emits correctness JSON, Qwen MTP commands inspect metadata and run synthetic harnesses |
| Loader helpers | `loader/config`, `loader/tokenizer`, `loader/safetensors`, `loader/weights`, `loader/gguf` | Config JSON, tokenizer JSON, mmap safetensors, sharded/single-file weight sources, and GGUF inspection/tokenizer helpers; safetensors metadata, nil helpers, deterministic names, checked eager totals, partial sharded-open cleanup, tokenizer merge helpers, and GGUF REAP/TurboQuant readiness fields are guarded |
| Placement policy | `backends/placement` | Backend-neutral budget manager and layer placement estimator; device memory availability is caller-supplied; accounting rejects invalid categories and estimator math is saturating |
| SIMD backend | `backends/simd/runtime`, `backends/simd/kernels` | Package name remains `simd` for runtime imports; scalar/asm dispatch wrappers live in `runtime`; reusable CPU kernel bodies live in `kernels`; BF16/Q4/NVFP4 are grouped under runtime quant subpackages; scalar fallbacks, BF16 GEMV, empty-slice dispatch, per-call GEBP scratch, and SGEMM/GEBP/gather byte offsets are bounds/overflow-guarded |
| Half-precision conversion | `half` | Stdlib-only leaf package for IEEE-754 FP16 and bfloat16 → float32 (`F16ToF32`, `BF16ToF32`); consolidated from previously-duplicated copies in `loader/gguf`, `model`, and `model/ideogram4`, proven bit-equivalent across all 65536 inputs (only NaN payloads differed). Importable anywhere without cycle risk |
| Vulkan backend | `backends/vulkan` | Vulkan loader/device/buffer/shader dispatch scaffolding and embedded SPIR-V assets; diagnostics are opt-in via `GO_PHERENCE_VULKAN_DEBUG` |
| BERT/GTE | `models/bert` | Encoder path split out of the decoder package |
| KV runtime | `runtime/kv` | TurboQuant state, compressed KV cache, shared KV byte estimator, and staging/rollback helpers with layout, accessor, memory-accounting, protected-layer input, and overflow guards |
| Memory runtime | `runtime/memory` | mmap residency advice/range tracking used by safetensors eager loading and future streaming; nil advisors are inert and malformed tracked ranges are sanitized with saturating accounting |
| Quant compatibility | `runtime/quant` compatibility wrappers | Compatibility API that delegates MLX to `backends/mlx`, Q4/GPTQ to `backends/simd/quant/q4`, and NVFP4 to `backends/simd/quant/nvfp4`; new backend-owned code should import those packages directly |
| Model packages | `model`, `model/qwen`, `model/gemma4`, `model/llama` | Shared LLaMA-family loader/forward and generation scaffolding remain in `model`; Qwen3.5/Qwen3.6 and native MTP live in `model/qwen`; Gemma4 diagnostics live in `model/gemma4`; LLaMA-specific primitives that can be split safely live in `model/llama`; helper guards cover MTP drafter/verifier/acceptance/KV commit edges, speculative proposer/config/stats/checkpoint paths, CPU decode finish/final norm, generation allocation setup, MoE, GGUF REAP/TurboQuant planning, inference helpers, CPU forward-layer entrypoints, KV sizing, GPU prefill/LM-head, GEMV, GQA arithmetic, and opt-in loader/prefill logging |
| NVIDIA backend | `backends/nvidia/runtime`, `backends/nvidia/ptx`, `backends/nvidia/ioctl` | NVIDIA runtime dispatch, driver loading, DevBuf/stream/stats/module handling, GPU-resident expert cache, and quantized runtime wrappers live under `runtime`; raw PTX strings live under `ptx` with BF16/Q4/MLX/NVFP4 grouped by quantization; direct ioctl experiments live under `ioctl` |
| Tensor graph | `tensor` | Lazy tensor DAG/runtime; direct import of `backends/simd/runtime`; malformed-input validation across shapes, unsafe views, broadcasting, realization, rewrite/fusion, NN/convenience helpers, matmul/linear, and modules |


## Shared Runtime Hardening Baseline

The recent audit now treats guard behavior in shared packages as part of the architecture, not incidental cleanup:

- `tensor` constructors and shape helpers reject negative or overflowing dimensions before allocation; malformed shape sizing/contiguity/broadcast checks fail before indexing or allocating.
- Tensor entrypoints are nil-safe or explicitly panic with domain errors before dereferencing internal fields.
- Realization, broadcast, reduction, rewrite, and fusion paths validate malformed UOps, source lists, buffer lengths, and reduction metadata before indexing.
- Unsafe float32 views, embedding, matmul, linear, softmax, layernorm, GELU, convenience ops, and module wrappers validate dimensions, backing-data lengths, and optional parameters before slicing or dispatching SIMD kernels.
- `runtime/quant` compatibility wrappers, `runtime/kv`, `runtime/memory`, loader helpers, SIMD wrappers, and NVIDIA runtime dispatch wrappers follow the same policy: validate dimensions/pointers/layouts and checked arithmetic at API boundaries, then either return an error/nil/no-op or panic with a local diagnostic rather than relying on incidental index panics.
- SIMD scalar fallbacks bound all participating slices; scalar RMSNorm uses precise `math.Sqrt`; BF16 GEMV validates shape-product overflow; empty vector/BF16 calls route through scalar fallbacks; GEBP packing scratch is per-call; SGEMM/GEBP/gather wrappers validate dimensions, pointers, strides, CPU capability gates, checked float32 byte offsets, and overflow before unsafe slicing or pointer arithmetic.
- Safetensors validates known dtype byte sizes against declared shapes and data offsets at open time; nil file/sharded helpers are safe, tensor names are sorted deterministically, sharded eager-load totals are checked, partial sharded opens clean up already-open shards, tokenizer byte maps use one-time initialization for concurrent callers, and malformed BPE merges are rejected.
- Transitional model helpers validate staged MTP acceptance consistency, model-aware verifier token/position/logit/activation dimensions, shared-KV verifier sources, alias-safe MTP drafter projection products, q-only external-KV/layer dimensions, bounded multi-draft counts, speculative stats/rollback paths, CPU decode final norm/LM-head dimensions, generation allocation setup, MoE loader/forward edge cases, model-specific KV dimensions, embedding/LM-head/per-layer-input backing slices, CPU forward-layer entrypoints, GPU prefill/chunked-LM-head entrypoints, and low-level GEMV/GQA product arithmetic before slicing or dispatch.
- Transitional NVIDIA runtime helpers validate `DevBuf` receiver/upload state, graph launches, stream kernel arguments, allocation sizes, Q4/MLX packed-weight layouts, expert-pool IDs, experimental NV helper inputs, dense SGEMM/LM-head buffers, JIT specs, BF16 buffers, and RoPE/attention tensor shapes before driver calls or kernel dispatch; NVIDIA progress logs are opt-in under `GO_PHERENCE_GPU_DEBUG`.

Later package moves should preserve this policy and keep focused regression tests close to the package that owns the guard.

## Backend Coverage and Validation References

Detailed backend ownership, parity, and validation tracking lives in focused documents rather than this architecture overview:

- [backend-layout.md](backend-layout.md) — package ownership and source-tree boundaries.
- [kernel-coverage.md](kernel-coverage.md) — primitive/backend implementation matrix.
- [backend-parity-matrix.md](backend-parity-matrix.md) — scalar/reference owners and hardware-gated parity targets.
- [malformed-input-coverage.md](malformed-input-coverage.md) — exported wrapper malformed-input coverage index.
- [validation-gates.md](validation-gates.md) — phase-level test, vet, CPU gate, and benchmark commands.
- [benchmark-snapshot-queue.md](benchmark-snapshot-queue.md) — hot-path benchmark entrypoints and refreshed snapshot status.
- [vulkan-dispatch-inventory.md](vulkan-dispatch-inventory.md) and [vulkan-validation-plan.md](vulkan-validation-plan.md) — Vulkan wrapper/pipeline status.
- [nvidia-quant-boundaries.md](nvidia-quant-boundaries.md), [bf16-parity.md](bf16-parity.md), and [nvfp4.md](nvfp4.md) — NVIDIA/quantized runtime boundaries.

## Weight Format Pipeline

```
HuggingFace (mlx-community, GPTQ, BF16)
    │
    ▼
loader/safetensors + loader/weights (GetFloat32, GetBF16, GetInt32, GetRaw)
    │
    ├─── MLX 4-bit: backends/mlx.LoadWeight validates packed shape + F32/F16/BF16 scales/biases
    │    └─── GPU: transpose → GPTQ kernel + bias correction
    │
    ├─── GPTQ 4-bit: loader reads qweight/g_idx/scales/qzeros → backends/simd/quant/q4 validates before dequant or GemvSym
    │    └─── GPU: direct tiled GEMV
    │
    └─── BF16/F16/F32: load → tensor (optional BF16 native path)
         └─── GPU: DevBuf upload
```

## Model Architecture Support

| Feature | llama | qwen2 | qwen3 | gemma3 | gemma4 |
|---|---|---|---|---|---|
| RoPE | ✅ | ✅ | ✅ | ✅ dual | ✅ dual |
| GQA | ✅ | ✅ | ✅ | ✅ | ✅ |
| QK-Norm | — | — | ✅ | ✅ | ✅ |
| 4-norm residual | — | — | — | ✅ | ✅ |
| Sliding window | — | — | — | ✅ | ✅ |
| Embed scaling | — | — | — | ✅ ×√h | ✅ ×√h |
| Norm +1 offset | — | — | — | ✅ | ✅ |
| GELU activation | — | — | — | ✅ | ✅ |
| BOS token | — | — | — | ✅ | ✅ |
| Tensor prefix | — | — | — | — | ✅ language_model. |
| Q/K/V bias | — | ✅ | — | — | — |
| head_dim ≠ h/heads | — | — | ✅ | ✅ | ✅ |
| MTP drafter assets | — | — | research | — | internal |

## Speculative Decoding / MTP

There are two distinct speculative tracks:

1. **Gemma4/Qwen MTP internals** — custom drafter/checkpoint assets, still disabled in public generation.
2. **Stock-weight speculative scaffold** — Orthrus-inspired verifier mechanics without custom weights, opt-in on the CPU backend via `--speculative`.

### MTP internals

Gemma4 MTP support now has internal verifier/drafter integration pieces, but it remains deliberately disabled in public generation/CLI paths. Implemented pieces:

- `LoadGemma4MTPDrafter` for `gemma4_assistant` safetensors assets with q-only attention blocks.
- Assistant projection helpers: token embedding row copy, masked ordering lookup, `PreProjectInto`, and `PostProjectInto`.
- Main-model verifier primitives: raw/scaled token embeddings, Gemma4 per-layer input preparation, CPU decode finish helper with copied final activation, LM-head logits, and greedy argmax.
- Initial CPU verifier loop: `RunMTPVerifierForward` validates plan/KV contracts, rejects unsupported Gemma4 PLI/batched semantics, runs real CPU layers through `ForwardLayer`, stages float KV, and returns per-position logits plus final activation.
- Acceptance helpers: `AcceptMTPDraft`, `AcceptMTPDraftFromLogits`, LiteRT-style bonus-token accounting, and `MTPAcceptance.Validate` before staged KV commits.
- `runtime/kv` staging helpers for candidate rollback/commit in both uncompressed and TurboQuant-backed caches; model-aware verifier plans/results validate vocab/token/position/logit/activation dimensions before deriving acceptance.
- Internal drafter/verifier seams: projection-only, synthetic q-only, and local real-asset contract tests can run against an explicit external-KV view; bounded multi-step drafter and multi-draft speculative helpers record LiteRT-style stats and restore staged verifier KV on verifier/stat errors.

Remaining MTP architecture work is full Gemma4 PLI/batched verifier semantics, production q-only drafter parity against real assistant assets, adaptive draft-count policy, GPU/hybrid support, and public generation wiring after CPU/GPU smokes.

### Stock-weight speculative scaffold

The stock-weight path deliberately avoids Orthrus custom `*_diff` tensors and instead provides a reusable verification/proposer scaffold for normal Qwen/Gemma/LLaMA weights:

- `GenerateSpeculative` / `GenerateSpeculativeWithStats` are opt-in CPU entrypoints used by `llmgen`, `llmchat`, and `llmserver` when `--speculative` is set.
- `SpeculativeProposer` is pluggable. Current proposers are `prompt` (prompt/suffix lookup), `repeat-last` (cheap verifier stress), and `none` (fallback overhead baseline).
- `CPUDecodeState` owns output/KV checkpoint, restore, `GenerateGreedy`/`DecodeOneGreedy`, accepted-prefix commit, and `VerifyGreedyBlock` contracts.
- Current verifier backend is `replay`: exact greedy verification by replaying the prepared CPU prompt. It is a correctness/measurement scaffold and can be slower.
- The `kv` backend selector is accepted but falls back to `replay` until a stateful KV-reusing verifier replaces the replay body.
- `SpeculativeStats` records backend, proposer, proposal/acceptance/fallback counters, emitted-token counts, tokens/step, average proposal length, plus reusable add/average helpers for benchmarks.
- `cmd/llm/specbench` compares normal vs speculative output, validates parity, supports prompt-file workloads and repeat averaging, and emits CSV suitable for tracking `backend=replay` to future `backend=kv` improvements.

## BF16 Pipeline

```
loader/safetensors BF16 → GetBF16() → []uint16 (zero conversion)
    │
    ├─── CPU: `backends/simd/runtime` package: BF16DotAsm (AVX2 445ns / NEON 8-wide)
    │         BF16RMSNormAsm (AVX2 1.4µs)
    │         BF16VecAddAsm (AVX2/NEON 8-wide)
    │
    ├─── GPU NVIDIA: ld.global.b16 + cvt.f32.bf16 (native Ampere+)
    │              ld.global.u16 + shl (emulated sm_80)
    │
    └─── GPU Vulkan: uint16 load + bitshift (universal)
```

## Kernel / Shader Inventory

| Backend | Current status |
|---|---|
| NVIDIA PTX | Hand-written kernels across GEMV/GEMM, attention/RoPE, norms, activations, BF16, NVFP4 dequant fallback, fused add-scaled accumulation, and utility paths; source assets live in `backends/nvidia/ptx` with quantized kernels grouped under `bf16`, `q4`, `mlx`, and `nvfp4`; dispatch/resource ownership lives in `backends/nvidia/runtime` |
| Vulkan SPIR-V | `backends/vulkan` owns shader assets for vector add, RMSNorm, GEMV, SiLU, attention score, RMSNormNoScale, RoPEPartial, and GELU paths; full forward dispatch is still pending |
| AVX2 asm | Runtime-gated vector, norm, dot/Saxpy, BF16, and SGEMM helpers in `backends/simd/runtime` with scalar fallback |
| NEON asm | Runtime-gated vector, norm, dot/Saxpy, BF16, and SGEMM helpers in `backends/simd/runtime` with scalar fallback; hardware verification still pending |
| Go scalar | Universal fallback for unsupported architectures or uncovered kernels |
