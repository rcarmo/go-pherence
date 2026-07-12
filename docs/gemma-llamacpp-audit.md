# Gemma-family inference graph audit against the local llama.cpp fork

Date: 2026-07-12

Go repository: `go-pherence`

Reference: local `llama.cpp` fork at `4a6735f1c` (`pr-24423`)

## Executive summary

The **Gemma4 text graph is structurally close to the fork**: per-layer attention type, shared-KV tail layers, Q/K normalization, no-scale V normalization, K=V fallback, proportional RoPE, post-attention/post-FFN norms, PLI injection, layer output scale, final softcap, and suppress-token bias are represented and have focused tests.

The implementation is **not yet graph-equivalent across the whole Gemma family**. Five correctness gaps remain material:

1. **P0 — positions above 2047 silently lose RoPE across the generic Gemma and Gemma4 paths.** CPU, MTP, and GPU consume tables hard-capped at 2048 although models advertise longer contexts.
2. **P1 — Gemma2 is incomplete:** attention softcap is absent and the 27B attention scale cannot be represented.
3. **P1 — Gemma3 27B attention scaling is wrong:** Go uses the generic head-dimension formula instead of llama.cpp's hidden/head-count formula.
4. **P1 — Gemma4 MTP strict selected-logit parity remains red/experimental.** The Q-only assistant graph exists, but batched verifier execution is env-gated and public generation still documents an external-KV handoff failure.
5. **P1 — DiffusionGemma is structurally aligned but numerically not closed.** Checked-in fixtures explicitly preserve layer-12/layer-29 router and output divergence; full-canvas top-token agreement is not full top-k/logit parity.

The runtime knobs are also not 1:1. Go covers the important DiffusionGemma entropy-bound values and has its own residency/cache controls, but it does not expose the fork's full MoE placement, FlashAttention, backend split/offload, diffusion algorithm/block/CFG/Gumbel, or `auto|on|off` device-KV/sampling controls through equivalent interfaces.

## Ranked findings

### P0 — generic Gemma and Gemma4 RoPE caches truncate valid context

**Go evidence**

- `model/rope.go:19-24` caps generic RoPE to 2048.
- `model/rope.go:48-52` caps both Gemma4 SWA and full-attention tables to 2048.
- Non-Gemma4 execution indexes the generic table by the real position in `model/forward_layer.go`, `model/mtp_prompt_context.go`, and `model/mtp_verifier_projection.go`.
- Gemma4 execution does the same in `model/forward_layer.go:148-161`, `model/mtp_prompt_context.go:263-276`, `model/mtp_verifier_projection.go:198-211`, and `model/gpu_forward.go:1127-1153`.
- `model/gpu_forward.go:498-512` uploads only the truncated Gemma4 tables.

The apply helpers cannot manufacture missing rows, so positions `>=2048` do not receive the advertised rotation. llama.cpp's Gemma1/2/3/4 graphs call `ggml_rope_ext` for requested positions and are not limited by this cache.

**Required fix:** replace fixed full-context tables with position-local/on-demand RoPE on CPU and GPU, or implement safely growing caches that preserve real GGUF factors and refresh device buffers. Add positions 2047/2048/4095 CPU↔llama and GPU↔CPU gates for generic and Gemma4 paths.

### P1 — Gemma2 attention softcap and 27B scale are unsupported

**llama.cpp reference**

- `src/models/gemma2.cpp:9,16` enables and loads attention softcap.
- `src/llama-graph.cpp:2080-2081,2126-2132` applies the fused or explicit `tanh(kq/cap)*cap` operation before softmax.

**Go evidence**

- `model/common/config.go:6-50` has `FinalLogitSoftcapping` but no attention softcap.
- `model/llama.go:1239-1242` only distinguishes Gemma4 scale `1.0` from generic `1/sqrt(head_dim)`; it has no Gemma2 QK softcap.
- llama.cpp's Gemma2 27B branch uses `1/sqrt(n_embd/n_head)`, which Go also cannot select.
- No dedicated Gemma2 GGUF graph loader exists alongside `model/gemma4_gguf_loader.go`.

This is classified P1 rather than P0 because the repository does not currently establish Gemma2 as a fully supported parity target; it is an unsupported/incomplete family member rather than a regression in the primary Gemma4 path.

**Required fix:** add attention-softcap metadata/config, apply it before attention softmax in CPU/SIMD/GPU paths, add model-size-specific attention scaling, and add a scalar formula oracle plus a real Gemma2 row fixture. Until then Gemma2 must not be advertised as llama.cpp-parity capable.

### P1 — Gemma4 MTP is structurally present but not production-equivalent

**Aligned**

- Q-only assistant consumes mapped target K/V in `model/mtp_drafter_loop.go` (`RunMTPDrafterStepWithExternalKV`, `runMTPDrafterQOnlyLayer`, and `drafterGQAAttention`), matching `src/models/gemma4-assistant.cpp:143-154`.
- Target embedding + hidden concatenation and pre/post projections are represented.
- Shared external KV maps SWA/full assistant layers to the target tail: `model/mtp_external_kv.go:5-20`.

**Gaps**

- True batched verifier layer lowering is gated by `GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS`: `model/mtp_verifier_batch_forward.go:454-456`.
- `docs/gemma4-mtp-benchmarks.md` and `docs/mtp-speculative.md` still record selected-logit mismatch and an external-KV handoff failure in the public CLI path.
- The same >2047 RoPE defect affects assistant and verifier paths.

**Required fix:** close selected-logit parity against the fork fixture first; repair external-KV prompt handoff; promote batched verifier only after real-model acceptance/logit/KV commit gates pass.

### P1 — DiffusionGemma numerical parity remains open despite structural alignment

**Aligned graph**

- Region-aware prompt/canvas embedding and masks.
- Q/K RMSNorm, no-scale V RMSNorm, proportional RoPE, prompt-KV prefill/decode split.
- Shared dense MLP + routed MoE, region-specific layer scales.
- Final softcap, device sampling, and previous raw-logit self-conditioning state.
- Structural op-order gates in `model/diffusiongemma/backend_graph_audit_test.go` and phase-aligned fixtures.

**Known numerical gaps**

- `model/diffusiongemma/testdata/gguf_hi_1x1_parity_status.json` and `llamacpp_reference_fixture_test.go:288-296` deliberately lock current layer-12/layer-29 divergence rather than parity.
- The slow full-canvas fixture agrees on the first top token but permits a bounded logit difference; it is not exact top-k parity.
- `model/diffusiongemma/config.go:91-92`, `model/diffusiongemma/model.go:10-11`, and `reference_gaps.go` correctly keep global runtime/reference readiness false.
- Vision tower/insertion is not reference-complete.

**Required fix:** compare identical first-decode tensors at each op boundary, replace divergence fixtures with strict parity fixtures as each boundary closes, then promote full-canvas top-k/logit/entropy and generated response to required gates.

### P1 — Gemma3 27B attention scaling is incorrect; broader coverage is partial

Go has explicit Gemma3 branches for `(1 + weight)` norms and BF16 rounding (`model/llama.go:614-629,999-1003,1099-1105,1289-1293,1351-1368,1394-1407`). llama.cpp's Gemma3 graph order—embedding scale, q/k norms, hybrid SWA, post norms, final softcap—is broadly represented by the generic path.

However, llama.cpp selects `1/sqrt(n_embd/n_head)` for Gemma3 27B and `1/sqrt(head_dim)` otherwise (`src/models/gemma3.cpp`). Go's `model/llama.go:1239-1242` only distinguishes Gemma4 from one generic formula, so the 27B graph is wrong. There is also no complete real-model llama.cpp graph fixture comparable to the Gemma4/DiffusionGemma work.

**Required fix:** add the model-size-specific attention scale and real GGUF per-op/final-logit fixtures for SWA and global layers, including BF16/quantized matmul boundaries.

### Gemma1 support contract — explicitly unsupported

The fork's Gemma1 graph (`src/models/gemma.cpp`) has no q/k norms, no post-attention/post-FFN norms, ordinary causal KV, GELU-parallel FFN, tied output, and no softcaps. Go can express much of this through optional tensors, but there is no dedicated loader/parity gate proving exact Gemma1 behavior.

`LoadLlama` therefore rejects the canonical Gemma1 `model_type: "gemma"` before opening weights. Gemma1 must not be advertised as supported until a dedicated graph and real-model parity gate replace that rejection. Gemma2, Gemma3, and Gemma4 use distinct model types and are unaffected.

## Structural parity matrix

| Feature | Gemma4 Go vs fork | Evidence / note |
|---|---|---|
| Token embedding scale | Match | `sqrt(n_embd)` in both; multimodal/raw embeddings are handled separately. |
| SWA/global per-layer attention | Match | `LayerTypes`, local/global head dimensions and window handling. |
| Shared-KV tail | Match | `HasKV=false` + `KVSourceLayer`; tail computes Q and reuses earlier KV. |
| Q/K norm | Match | Per-head RMSNorm before RoPE. |
| V norm | Match | no-scale RMSNorm; K projection may seed distinct K-normalized and V-no-scale branches. |
| RoPE formula/factors | Match <=2047; **gap above** | GGUF factors and theta progression are tested, but cache is truncated. |
| Attention scale | Match for Gemma4 | `1.0`, unlike Gemma2/3. |
| Post-attention norm + residual | Match | Implemented in CPU and GPU graph paths. |
| Dense FFN activation | Match | ggml tanh-GELU gated parallel MLP. |
| Gemma4 routed MoE | Partial | DiffusionGemma router formula/top-k/down scales are aligned; ordinary Gemma4 MoE real-model coverage is thinner. |
| PLI injection | Match structurally | gate × per-layer input, projection, RMSNorm, residual, output scale. |
| Final norm/head/softcap | Match | tied/untied handling, `tanh(x/c)*c`. |
| Suppress-token bias | Match | metadata loaded; logits set to `-Inf`. |

## Tuning/knob comparison

### Equivalent or substantially covered

| llama.cpp fork | Go equivalent | Status |
|---|---|---|
| Diffusion EB `t_min/t_max` | `diffusiongemmarun -t-min/-t-max` | Match |
| EB entropy/stability/confidence/max steps | `-entropy-bound/-stability/-confidence/-denoise-steps` | Match |
| Prompt-KV reuse | default incremental KV plus `GO_PHERENCE_DIFFUSIONGEMMA_DISABLE_INCREMENTAL_KV` / `REQUIRE_INCREMENTAL_KV` | Functionally covered, interface differs |
| GPU sampling | device sampler implementation plus env/backend graph controls | Functionally covered, no identical `auto|on|off` CLI |
| CPU reference path | `-cpu-dispatcher -allow-slow-cpu` | Covered as explicit oracle |
| Context/KV compression | `-ctx-size`, TurboQuant cache types in generic server/smoke | Go-specific; not llama.cpp quant-cache parity |

### Missing or materially different

| llama.cpp fork knob | Go state |
|---|---|
| `--cpu-moe`, `--n-cpu-moe` and speculative-draft variants | No equivalent common layer-placement contract; DiffusionGemma has model-specific expert residency/env controls and `diffusiongemmaserve -cpu-experts`. |
| `--flash-attn` | No equivalent user-facing selection/compatibility contract for Gemma; Go internally tries optimized paths. |
| `--cache-type-k/v` llama.cpp formats | Go exposes TurboQuant (`turbo4/turbo2`, etc.), which is a different implementation and must not be called bit/numeric equivalent. |
| `-ngl`, split mode, tensor split, main GPU, backend offload | Go has coarse `-gpu-layers`/GPU dispatcher and model-specific residency, not equivalent backend placement semantics. |
| `--diffusion-blocks`, algorithm 0..4, algorithm temperature | Go DiffusionGemma implements its entropy-bound generation path, not the fork's whole generic diffusion algorithm surface. |
| diffusion CFG scale / Gumbel noise / epsilon | Not exposed by `diffusiongemmarun`; not part of the currently aligned DiffusionGemma EB path. |
| `--diffusion-kv-cache auto|on|off` | env-based enable/require/disable behavior, not identical tri-state CLI. |
| `--diffusion-gpu-sampling` and `--diffusion-gpu-sample-reduce` tri-state | implementation exists, interface/default negotiation differs. |
| Gemma convenience embedding/vision presets | No direct equivalents. |

## Verification gates already worth keeping

- Gemma4 real GGUF Q/K/V, K=V, PLI, GGML/BF16, tail softcap/suppress-token and shared-KV tests under `model/*gemma4*`, `pli_*`, `mtp_*`.
- DiffusionGemma `make diffusiongemma-golden-gate` for llama fixtures, scalar quantized-dot oracles, structural trace gates, and bounded GPU↔CPU smoke.
- `TestQ6I8DotAVX2MatchesScalar` and Q4_K/Q5_0/Q8_0 expert scalar oracles.

These are necessary but not sufficient: several current fixtures document a stable mismatch rather than enforce equality.

## Recommended closure order

1. Replace the generic and Gemma4 fixed 2048 RoPE tables with correct on-demand CPU/GPU RoPE and add long-context gates.
2. Add Gemma2 attention-softcap metadata and model-size-specific attention scaling, or explicitly reject Gemma2.
3. Correct Gemma3 27B attention scaling and add SWA/global real-model fixtures.
4. Close Gemma4 MTP selected-logit and external-KV handoff parity, then enable batched verifier by default.
5. Continue DiffusionGemma first-difference closure from the current layer-12/layer-29 fixtures; promote exact full-canvas top-k/logit/entropy gates.
6. Add real Gemma1 fixtures before claiming family-wide parity.
7. Normalize user-facing knobs only after graph correctness: introduce a documented common placement/sampling interface and map unsupported llama.cpp knobs to explicit errors rather than silently ignoring them.
