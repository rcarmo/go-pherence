# MTP (Multi-Token Prediction) Speculative Decoding

## Overview

MTP is a speculative decoding technique where a small **drafter model** predicts
several candidate tokens, and the large **verifier model** validates them in one
batched forward pass. If the drafter's predictions match, multiple tokens are
accepted per step — up to 3× speedup.

This document covers the custom-drafter/MTP track. The separate stock-weight
speculative scaffold inspired by Orthrus lives in [orthrus.md](orthrus.md): it
uses normal model weights, pluggable cheap proposers, structured stats, and the
`cmd/llm/specbench` CSV harness, but currently verifies with `backend=replay` until
a KV-reusing verifier block is implemented.

## Architecture

### Drafter models

Local E2B BF16 asset: `models/gemma4-e2b-mtp-drafter`.

Local 31B MLX 4-bit assistant asset: `models/gemma4-31b-it-mtp-assistant-4bit`.

- Top-level `model_type: gemma4_assistant`, `architectures: [Gemma4AssistantForCausalLM]`
- Nested text config is `model_type: gemma4_text`
- `hidden_size: 256` (vs 1536 in main model)
- `num_hidden_layers: 4` (vs 35 in main model)
- `intermediate_size: 2048`
- `num_attention_heads: 4`, `num_key_value_heads: 1`
- layer types: 3 sliding layers (`head_dim=256`) + 1 full layer (`global_head_dim=512`)
- `num_kv_shared_layers: 4`; all drafter layers consume external/shared KV rather than owning K/V projections
- Disk: 151 MB BF16 safetensors + 31 MB tokenizer
- VRAM estimate: ~50 MB plus runtime buffers

### Special tensors in local E2B safetensors
- `pre_projection.weight`: `[256, 3072]` — maps `embedding(prev_token)[1536] || activation[1536]` into drafter hidden size 256.
- `post_projection.weight`: `[1536, 256]` — maps drafter hidden/projected state back to the main hidden size for the next drafter step and verifier handoff.
- `masked_embedding.centroids.weight`: `[2048, 256]` — centroid embedding table.
- `masked_embedding.token_ordering`: `[262144]` — vocab → centroid ordering/index data.
- `model.embed_tokens.weight`: `[262144, 256]` — drafter-token embedding table.
- Per-layer tensors include only `q_proj`, `q_norm`, `o_proj`, MLP weights, norms, and `layer_scalar`.
- **No `k_proj`, `v_proj`, `k_norm`, or `v_norm` tensors exist in the drafter**; those must come from shared/base-model KV state.

Current implementation status (after the backend/model reorganization; Qwen-specific code now lives in `model/qwen`, Gemma4 diagnostics in `model/gemma4`, and generation extraction remains future work):
- `LoadGemma4MTPDrafter` loads the local assistant assets into a dedicated q-only drafter structure, including exact-shape/config validation for `pre_projection`, `post_projection`, optional masked embedding tensors, and all four q-only layers.
- The 31B assistant path keeps large MLX 4-bit matrices packed (`EmbedTokensMLX`, projection weights, attention weights, and MLP weights) and dispatches q-only drafter GEMVs through `backends/mlx` on-the-fly compute instead of expanding the assistant to F32. Token embedding lookup dequantizes only the requested row.
- Helper methods cover assistant token row copies, masked embedding ordering lookups, alias-safe `PreProjectInto`, and alias-safe `PostProjectInto`.
- Main-model helper primitives expose raw/scaled token embeddings, Gemma4 per-layer input preparation, a CPU decode finish helper that returns copied final activations, LM-head logits, and greedy argmax outside `Generate`; `Generate` now uses these shared helpers, and prompt-context KV import checks per-layer KV length overflow before seeding graph decode state.
- `runtime/kv` staged KV helpers can checkpoint, restore, and keep only the accepted prefix plus verifier bonus token for both uncompressed and TurboQuant-backed KV caches.
- `AcceptMTPDraft`/`AcceptMTPDraftFromLogits` encode LiteRT-style accepted-prefix plus bonus-token semantics. `VerifiedCount` deliberately excludes the bonus token to match LiteRT-LM accounting, and `MTPAcceptance.Validate` rejects inconsistent manually assembled acceptance state before KV commit.
- `MTPVerifierPlan` prepares `[input_token]+drafted` token IDs and absolute verifier positions with model-aware token/vocab and overflow checks. `MTPVerifierBatchInputs` materializes the verifier graph inputs as a batch bundle: scaled embeddings plus optional Gemma4 per-layer inputs for every `[input]+drafted` row, backed by checked contiguous flat hidden/PLI buffers for SIMD/GPU lowering; batch-forward validation now rejects stale hidden/PLI flat buffers, missing per-row PLI slots, or scratch plans that no longer match row views/model shapes. It carries `MTPVerifierAttentionPlan` for per-layer causal/sliding KV windows and `MTPVerifierBatchScratchPlan` for backend-neutral per-layer scratch/shape allocation; attention plans are validated against the exact model-derived full/sliding ranges rather than merely range-checked. `ProjectMTPVerifierLayerQKVBatch` now lowers the first nonzero-layer slice over flat verifier hidden rows: input norm, Q/K/V projection, BF16 rounding, K=V handling, q/k/v norms, and RoPE with exact row-parity tests. The gated full-layer verifier batch scaffold can now use dense, MLX, or quantized/QAT projection weights via a row-wise `mvQ` oracle batch helper for Q/K/V/O/Gate/Up/Down while true quantized batch kernels are pending; it also supports Gemma4 PLI gate/projection/post-norm over the verifier batch. The verifier row-loop oracle supports the same QuantWeight projections through `ForwardLayer`. `RunMTPVerifierBatchForward` is the single verifier execution entry point: it lowers tail/final-norm+LM-head logits with `FinishCPUDecodeBatch` and still lowers full nonzero layers through a sequential row/layer loop by default while these batched layer primitives are introduced. A complete nonzero-layer batch-lowering scaffold is guarded behind `GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS`; the current synthetic verifier parity test now matches the row-loop oracle, including layer-scalar semantics, but the path stays off by default until broader real-asset acceptance parity is covered.
- `MTPVerifierResult` validates verifier logits/activation outputs, derives acceptance, and can commit the accepted KV prefix for float or TurboQuant-backed caches; verifier-result float/model-aware commit helpers check verifier tokens, logits widths, activation widths, and recompute acceptance from logits before retaining KV so mutated or dimension-drifted result state cannot bypass LiteRT-style matching semantics. The real batch verifier path now records one final activation row per verifier token so MTP can seed the next drafter from the logits-derived accepted-prefix/bonus row instead of blindly using the last verifier row; `CommittedActivation()` also rejects mutated acceptance that disagrees with logits before selecting that row. K=V handling is aligned across load-time QAT/GPU binding, ordinary CPU/GPU generation and prefill, `ForwardLayer`, prompt-context prefill, and verifier-batch projection: omitted/shared V projections now copy K for `AttentionKEqV` rather than requiring or uploading a separate V matvec, including native QAT paths, bounded GPU work-buffer prefix copies, and hybrid GPU CPU-fallback QAT projections where applicable. Ordinary CPU generation, `ForwardLayer`, MTP prompt/drafter lowering, verifier-batch QAT projection helpers, MoE MLX router/expert projections, hybrid GPU CPU-fallback QAT projections, and ordinary GPU packed MLX/Q4 dispatch and compact MLX LM-head dispatch now check projection dimensions/failures instead of silently continuing with zeroed outputs; ordinary CPU decode and batched prefill scratch allocation also use checked attention, batch, and per-layer max Q/KV widths. MTP/prompt/CPU fallback paths, q-only drafter layers and prompt-context external-KV source mapping, ordinary CPU/TurboQuant generation, and ordinary GPU KV buffer/decode metadata also share the same Gemma4 effective head/KV-dim derivation, so full-attention layers use `GlobalHeadDim` plus `NumGlobalKVHeads` even in manually assembled graph tests where `HeadDimLocal` is not pre-populated. `CommitGraphFloatKV` and `CommitGraphCompressedKV` validate the explicit `MTPExecutionGraph` before retaining the accepted-prefix+bonus KV window, including verifier token identity, verifier position count, logits row coverage, activation-row coverage when present, and a logits-derived acceptance recomputation so manually mutated acceptance state cannot choose a different keep window. `CommitGraphFloatKV` and `CommitGraphCompressedKVForModel` additionally revalidate verifier token/logit/activation widths against the owning model. `CPUDecodeState.CommitGraphAccepted` is the production-facing bridge that commits graph-validated KV and appends exactly the graph-approved output tokens; it also requires the decode checkpoint cursor to match the graph start position and dispatches through the model-aware float or compressed graph commit path before mutating output/KV, so verifier result, graph, model metadata, and KV commit semantics cannot silently diverge. `NewMTPVerifierResultForModel` additionally checks token IDs against vocab size, logits rows against vocab width, and final activation against hidden size for the real verifier path; `NewMTPVerifierResultRowsForModel` checks the full verifier activation-row batch.
- `MTPAcceptance.KVKeepTokens` plus `CommitAccepted*KV` helpers apply accept/reject results directly to staged verifier KV caches; `runtime/kv` owns the generic staging state while `LayerKVDims` derives the correct per-layer widths for Gemma4 variable/shared KV layouts.
- Drafter layers mark `KVSourceLayer=-1` because their K/V source is external. `MTPDrafterExternalKV` is the explicit read-only main-model KV view for q-only drafter layers; validation checks top-level pre/post projection shapes, unique source mapping, source sequence width, q-only dense/MLX projection dimensions, MLP dimensions, packed MLX layout, and all required norms. `Gemma4MTPDrafter.LayerKVDim` exposes the same full/sliding effective dimension contract for CLI synthetic-smoke KV allocation in both `llmgen` and `cmd/models/gemma4mtpsmoke`, and for local real-asset drafter contract tests.
- `RunMTPDrafterStepWithExternalKV` runs the projection shell plus the current CPU q-only layer path: Gemma-aware BF16/FP32 norms, checked dense/MLX q projection, q norm, Gemma-scaled external GQA attention, checked dense/MLX output projection, residual/post-attention handling, pre/post-FFN norms, checked dense/MLX MLP, layer scalar, final drafter norm, post-projection, main LM-head logits, and next-state construction. Zero-layer projection-only fixtures remain supported.
- `RunMTPDrafterSteps` runs a bounded internal multi-step drafter-only loop, carrying copied state between draft steps and returning drafted tokens, logits, activations, and final state. The real-asset contract test now loads the local Gemma4 main and MTP drafter assets when present, builds a minimal external-KV view, and proves one q-only drafter step reaches correctly shaped outputs.
- `cmd/models/gemma4mtpsmoke` and `cmd/llm/llmgen -mtp-smoke` expose a runtime-facing smoke for the 31B path: load the main model on the on-the-fly 4-bit path, load the packed 4-bit assistant, build minimal external KV, run one q-only drafter step, and print timing/shape JSON. Both smoke outputs include `Gemma4MTPGraphCapabilities` plus the missing public-generation blockers so CLI users can distinguish production-safe graph pieces from gated verifier-batch paths. Latest local 31B minimal-KV smoke: main load `16.25s`, assistant load `0.26s`, drafter step `0.47s`, packed embedding/projection/layer weights all true.
- `BuildMTPPromptContext` and `llmgen -mtp-real-prompt` add the real prompt handoff: run the prompt through the Generate-equivalent Gemma4 per-layer-input path, capture final activation and float KV, map drafter layers onto compatible main-model KV source widths, and run the packed drafter against real verifier state. `NewMTPDrafterExternalKVFromPromptContext` now owns that width-based source mapping in the model package so smoke and `-mtp-generate` share the same validated external-KV contract. `GPUModel.BuildMTPPromptContext` does the same prompt capture through the hybrid GPU/CPU path and copies GPU-resident KV back for MTP. Prompt seeding now computes final activation only and skips prompt LM-head logits, since MTP needs the previous prompt token plus final activation/KV rather than next-token sampling. Latest local 31B short-prompt real-KV smoke: CPU/on-the-fly `299.06s`, safer hybrid GPU `213.76s` with `-gpu-layers 17 -gpu-kv-max-seq 256`, aggressive hybrid GPU about `200–205s` with `-gpu-layers 18 -gpu-kv-max-seq 64`.
- `MTPSpeculationStats` tracks LiteRT-style drafted/verified/bonus/output counters. `MTPExecutionGraph` now makes the llama.cpp/LiteRT backend graph explicit for each speculative cycle: hidden-state-conditioned drafter steps, verifier `[input]+drafted` token/position batch, and the acceptance-dependent KV keep window (`accepted_prefix + bonus`). `MTPExecutionGraph.Validate` checks the drafter step input chain, required external-KV presence for q-only drafter layers, consistent per-cycle activation/external-KV view across drafter steps, external-KV sequence cursor alignment with the graph start position, unique external-KV source layers, verifier `[input]+drafted` tokens, contiguous verifier positions, and KV keep bounds before commit planning, and `CommitPlan` rejects acceptance prefixes whose tokens do not match the graph's drafted-token prefix. `MTPAdaptiveDraftPolicy` chooses a bounded draft count from remaining output budget and observed acceptance rate. `RunMTPSpeculativeStep` provides the compatibility one-token drafter→verifier→stats seam, while `RunMTPMultiDraftSpeculativeStep` drafts multiple tokens before one verifier pass and returns the graph alongside the verifier plan/result. `CPUDecodeState.RunMTPGraphDecodeStep` is the internal production-cycle contract: choose/validate `G`, run drafter→verifier, graph-commit KV, append graph-approved output tokens, and seed the next drafter state from the last committed verifier output token plus the committed verifier activation row (`accepted_prefix`) rather than from the drafter-only final state or the verifier batch's final row. `GenerateMTPGraphFromPromptContext` exposes this graph cycle as a model-level API from a prefilled `MTPPromptContext`, refreshes the drafter external-KV view from the committed decode state before each graph cycle, materializing compressed/TurboQuant KV when needed, so later drafts see accepted verifier KV rather than stale prompt-only KV, and returns per-cycle summaries with drafted tokens, verifier tokens, accepted prefix, bonus token, output tokens, and committed positions. `llmgen -mtp-generate -mtp-drafter <dir>` now wires that graph API into an explicit experimental CPU verifier generation path; regular generation remains unchanged unless the flag is set. When the remaining budget is a single token, `GenerateMTPGraphFromPromptContext` falls back to ordinary greedy decode so the experimental path can still emit exactly the requested token count, reports `GraphOutputTokens` / `MTP graph output` separately from `GreedyTailTokens` / `MTP greedy tail` and `UsedCompressedKV` / `MTP compressed KV` so graph-cycle output and KV modality are not conflated with fallback output, and records `FinalStateOutputLen` so callers know the drafter state covers only prompt+graph tokens and not the greedy-tail fallback. `MTPGraphGenerationResult.Validate` enforces that per-cycle summaries, non-negative requested max-token budget with exact generated-token equality including zero-token requests, mandatory vocab/hidden-size metadata for non-empty outputs, model-aware token ID ranges, verifier input/output IDs, per-cycle output-stream cursor/input-token alignment, verifier position starts/contiguity, committed-position prefix, maximal accepted-prefix length, accepted draft-prefix output content, commit-plan outputs, graph output token order in the final stream, all-drafts-accepted state, seeded step/drafted/verified/bonus/output stats deltas (including zero-cycle cases), greedy tail tokens, mandatory final-state output cursor coverage for non-empty outputs, final-state previous-token/hidden-activation width, capability readiness/public-blocker consistency even for zero-token edge cases, and final output length reconcile before returning. `llmgen -mtp-generate` prints graph output, greedy-tail count, final-state coverage (`MTP state covers`), and one compact `MTP cycle` line per graph cycle with input token, drafted tokens, verifier input batch, verifier output IDs, accepted prefix, bonus, output tokens, committed positions, full verifier positions, and all-accepted state. `Gemma4MTPGraphCapabilities` reports which graph pieces are production-safe, which are gated, that experimental `llmgen -mtp-generate` wiring is ready, that per-cycle external-KV refresh and exact token-budget tail fallback are implemented, and why public/default generation is not yet marked ready; public readiness now also requires real-asset acceptance parity and the gated full-layer verifier batch path to be default-enabled. That report is included in experimental MTP smoke JSON and in the `GenerateMTPGraphFromPromptContext` result surfaced by `llmgen -mtp-generate`.
- Remaining gap: harden the experimental `llmgen -mtp-generate` path into default-ready public generation: capture real verifier activation/KV from normal decode without a separate prompt-context prefill, promote the sequential Gemma4 PLI verifier to true batched verifier execution, prove q-only drafter numerical parity against real assistant assets, and decide default enablement policy.

### Data flow

```
Main model decode step N:
  → hidden_state[1536]

Execution graph (`MTPExecutionGraph`):
  - drafter inputs chain previous token → draft[0] → draft[1] ...
  - verifier batch is [input_token] + drafted_tokens at contiguous positions
  - commit window is accepted_prefix_len + verifier bonus token

Drafter loop (K iterations):
  hidden_draft = pre_projection(embedding(prev_token) || activation) [256]
  for layer in drafter_layers:
    hidden_draft = q-only layer(hidden_draft, KV=external main-model KV)
  hidden_main = post_projection(hidden_draft)                         [1536]
  logits = LM_head(hidden_main)                   [vocab]
  draft_token = argmax(logits)
  carry projected activation and drafted token into the next iteration
  → candidate_tokens[K]

Verifier (current internal CPU path):
  Run main model sequentially over [input_token] + K draft tokens with staged float KV
  Return G+1 logits rows plus final activation/projected state
  Accept matching prefix, emit verifier bonus token, discard rejected suffix

Verifier (target batched path):
  Share Gemma4 PLI/batched semantics with Generate/Prefill
  Run [input_token] + K draft tokens in one prefill-style verifier pass
  Commit only accepted-prefix + bonus KV after acceptance
```

### Local fit notes

The Gemma4 E4B MLX pair is currently the best local development target:

- main: `mlx-community/gemma-4-E4B-it-4bit`, ~4.9GiB local, hidden 2560, 42 layers
- assistant: `mlx-community/gemma-4-E4B-it-assistant-bf16`, ~183MiB local, hidden 256, 4 layers
- RTX 3060 smoke: all 42/42 main layers resident, compact MLX LM head resident, ~5.0GiB VRAM free
- real-prompt MTP smoke: 76 prepared prompt tokens prefill in `3.41s`, drafter step `0.10s`

Use E4B for verifier/prefill/MTP algorithm development, then re-run 31B as the stress path.

### Performance gain
- Drafter: ~0.5ms per token (tiny model, 4 layers, dim=256)
- Verifier: ~5ms for K tokens batched (same as single-token decode)
- If 3/4 draft tokens accepted: 4 tokens for cost of ~7ms vs ~20ms sequential
- Net: ~2.8× throughput improvement

## Implementation plan

1. **Drafter loader** ✅ — parse `gemma4_assistant` top-level config, nested `text_config`, q-only attention blocks, `pre_projection`, `post_projection`, and masked embedding tensors.
2. **Main-model verifier helper primitives** ✅ — raw/scaled token embeddings, Gemma4 per-layer inputs, LM-head logits, and argmax are now reusable outside `Generate`.
3. **pre/post projection** ✅ — helper methods for `pre_projection(embedding(prev_token) || activation)` and `post_projection(hidden_draft)`.
4. **Accept/reject semantics** ✅ — `AcceptMTPDraft`/`AcceptMTPDraftFromLogits` keep the matching prefix and emit the verifier bonus token on mismatch/all-accepted completion.
5. **KV staging primitives** ✅ — checkpoint/restore/keep-prefix helpers support both uncompressed and TurboQuant-backed caches.
6. **KV cache sync primitive** ✅ — staged KV can keep `accepted_prefix_len + 1` verified positions and discard rejected candidate tails.
7. **Main-model verifier result contract** ✅ — `MTPVerifierTokens`/`MTPVerifierPlan`/`MTPVerifierResult` define `[input_token]+drafted`, verifier positions, logits rows, final activation, acceptance, and KV commit hooks; model-aware construction validates vocab/logit/activation dimensions.
8. **Verifier-forward contract** ✅ — `MTPVerifierBatchInputs` validates/materializes verifier graph inputs into contiguous flat hidden/PLI buffers, `MTPVerifierAttentionPlan` captures per-layer full/sliding causal KV ranges, and `MTPVerifierBatchScratchPlan` captures per-layer scratch/shape allocation for CPU/SIMD/GPU lowering, including source KV widths for shared-KV verifier layers. `ProjectMTPVerifierLayerQKVBatch` provides the first batched nonzero-layer primitive over flat verifier rows, and the gated batch layer scaffold now accepts dense/MLX/QAT projections by using row-wise `mvQ` as the quantized oracle where needed. It also covers Gemma4 PLI residuals in the same order as `forwardMTPPromptLayer`. `ForwardLayer` supports QuantWeight Q/K/V/O/MLP projections and rejects malformed MLX packed projection failures, keeping the row-loop oracle aligned with QAT verifier layers. `RunMTPVerifierBatchForward` is the single verifier execution entry point, with batched final-norm/LM-head tail lowering via `FinishCPUDecodeBatch`, checked attention KV range offsets in the gated full-layer lowering, and sequential nonzero-layer lowering over the batch contract while the remaining vectorized layer kernels land. The experimental full-layer batch path is gated off by default; its synthetic verifier parity now matches the row-loop oracle, including mixed Gemma4 sliding/full-attention per-layer KV widths and shared-KV layer reuse, and the next step is broader real-asset acceptance parity before enabling it generally.
9. **Initial CPU verifier path** ✅ — run a short sequential CPU forward over materialized `[input_token] + drafted_tokens`, return per-position logits and final activation, and stage candidate float KV updates; graph decode can stage compressed/TurboQuant decode states by materializing a float verifier shadow, verifying that the shadow prefix matches the compressed cache prefix, replaying staged verifier rows into compressed caches, and using the existing graph commit/rollback bridge, with regression coverage for nonzero-layer verifier KV rows; `GenerateMTPGraphFromPromptContext` and `llmgen -mtp-generate -turbo-quant` can seed compressed KV directly from prompt context. Non-PLI models use `ForwardLayer`; Gemma4 PLI models consume batched per-token `Gemma4PerLayerInputs` through `forwardMTPPromptLayer`, whose dense/quant/MLX projection paths now report packed-kernel failures instead of silently continuing, matching the prompt handoff path while the future batched verifier layer kernels are designed.
10. **Drafter forward loop** ✅/internal — projection-only, synthetic q-only, and local real-asset contract paths now run with explicit external KV, Gemma-aware norms/attention scale, q-only attention, MLP, final norm, post-projection, and main LM-head logits. `RunMTPDrafterSteps` carries state across bounded multi-step drafts. Remaining work is numerical parity against real assistant outputs and adaptive policy.
11. **End-to-end speculative decode** ✅/experimental CLI — `RunMTPSpeculativeStep` integrates one drafter step, verifier forward, stats, and an explicit `MTPExecutionGraph`; `RunMTPMultiDraftSpeculativeStep` does the same for multiple drafted tokens in one verifier pass. Graph-backed float/compressed KV commit helpers, `CPUDecodeState.CommitGraphAccepted`, `MTPAdaptiveDraftPolicy`, `CPUDecodeState.RunMTPGraphDecodeStep`, `GenerateMTPGraphFromPromptContext`, `Gemma4MTPGraphCapabilities`, and experimental `llmgen -mtp-generate` wiring are implemented. Experimental smoke CLIs report the capability surface, including compressed/TurboQuant verifier staging support for graph decode. Remaining work is hardening the experimental CLI path into default-ready public generation and completing real-asset verifier-batch parity/default enablement.
12. **31B packed smoke** ✅/experimental CLI — `cmd/models/gemma4mtpsmoke` and `llmgen -mtp-smoke` validate that the 31B main model and packed 4-bit assistant can execute one drafter step without dequantizing assistant matrices to F32.
13. **Real prompt activation/KV smoke** ✅/experimental CLI — `BuildMTPPromptContext` plus `llmgen -mtp-real-prompt` prove the main-model activation/KV handoff for Gemma4 31B, with CPU/on-the-fly prompt prefill still the dominant bottleneck.
14. **Adaptive K** — track acceptance rate by task/prompt class and adjust draft length.

## Reference Implementations

### Google LiteRT-LM — Gemma 4 single-position MTP

Sources inspected:
- <https://github.com/google-ai-edge/LiteRT-LM>
- `runtime/executor/llm_litert_mtp_drafter.{h,cc}`
- `runtime/executor/llm_litert_compiled_model_executor.cc`
- `schema/capabilities/speculative_decoding.cc`
- HF model cards: `litert-community/gemma-4-E2B-it-litert-lm`, `litert-community/gemma-4-E4B-it-litert-lm`

Key mechanics:
- MTP support is detected from the `.litertlm` bundle: presence of a TFLite model section with `model_type = "tf_lite_mtp_drafter"` marks the file as speculative-decoding-capable.
- The main model exposes a separate `verify` signature. LiteRT-LM infers the draft count from `verify.input_pos` shape: `num_draft_steps = len(input_pos) - 1`.
- The drafter model is compiled separately from the main prefill/decode model, with a `.mtp_drafter` compilation cache suffix.
- The first MTP call after prefill runs a normal single decode first, then passes that decode output's `activations` tensor into the drafter.
- Later MTP calls do not rerun a normal decode first; the drafter keeps enough internal state to restart from the last verifier activation.
- Drafter input `activations` is the concatenation of:
  - token embedding for the previous/generated token, looked up with `EmbeddingLookupManager.LookupDecode`, and
  - projected activations from either the previous drafter step or the verifier output.
  - LiteRT-LM comments describe this as `[B=1, T=1, D=3072]` for E2B/E4B-scale models: embedding `[1536]` + projected activation `[1536]`.
- Each drafter iteration runs the MTP drafter asynchronously, samples greedily from drafter `logits`, and carries forward `projected_activations` for the next drafter step.
- Verification is one batched main-model pass over `[input_token] + drafted_tokens`:
  - fills `input_pos` for `position..position+G`,
  - fills causal mask for `G+1` steps,
  - looks up normal embeddings and Gemma4 per-layer embeddings for all `G+1` tokens,
  - uses the main model's input KV cache and writes the output KV cache for verified/drafted positions.
- Verifier samples `G+1` token IDs from main logits. Accepted tokens are the matching prefix between verifier IDs and drafted IDs. On first mismatch, LiteRT-LM emits the verifier token as a **bonus token**; if all drafts match, it emits verifier ID at index `G` as the bonus token.
- Reported accounting excludes the bonus token: `num_drafted_tokens += G`, `num_verified_tokens += accepted_prefix_len`, and logs success rate on teardown.

Important difference vs the initial go-pherence plan:
- LiteRT-LM's drafter is **not just a standalone tiny autoregressive model**. It consumes the previous token embedding plus a projected activation vector, and the main model needs a verifier path that returns `activations`. This is closer to a hidden-state-conditioned MTP head than a generic auxiliary LM.
- For go-pherence, implement the verifier hidden/activation output first, then wire drafter state. Treat the drafter KV cache question carefully: LiteRT-LM passes base-model KV buffers into the drafter for drafting, and separately uses base-model `verify` to update output KV cache for accepted/drafted candidates.

Performance notes from public HF cards:
- `gemma-4-E2B-it.litertlm`: 2.59 GB bundle; text decoder weights 0.79 GB; embeddings 1.12 GB mmap'd. S26 Ultra GPU baseline 51.5 tok/s; speculative decoding task results: 66.5–91.7 tok/s. CPU baseline 40.7 tok/s; speculative task results: 36.3–47.5 tok/s.
- `gemma-4-E4B-it.litertlm`: 3.66 GB bundle; text decoder weights 2.24 GB; embeddings 0.67 GB mmap'd. S26 Ultra GPU baseline 21.9 tok/s; speculative task results: 36.7–49.4 tok/s. CPU baseline 17.0 tok/s; speculative task results: 21.1–29.5 tok/s.
- The HF cards note speculative decoding is task-dependent because drafter agreement varies with prompt/task.

Implementation implications for go-pherence:
1. Add a main-model `Verify(tokens []int)` path that runs a short batched forward, returns logits for each position, updates candidate KV, and exposes final hidden/projected activations. `MTPVerifierBatchInputs` plus `MTPVerifierAttentionPlan` now define that graph input and mask contract.
2. Preserve or compute Gemma4 per-layer embeddings for all verifier tokens; LiteRT-LM explicitly feeds `per_layer_embeddings` to verifier. `MTPVerifierBatchInputs` now materializes them for every verifier row; the remaining fidelity step is replacing the sequential layer loop with a true verifier-prefill style batched runner.
3. Model the drafter input as `embedding(prev_token) || activation`, not just token IDs. `PreProjectInto` and `PostProjectInto` implement the projection handoff pieces.
4. Apply acceptance accounting with accepted-prefix length plus bonus token, mirroring LiteRT-LM. `AcceptMTPDraftFromLogits` and `CommitAccepted*KV` are the tested bridge from verifier logits to KV-cache state.
5. Consider a `.litertlm` inspector later: section metadata can reveal whether an artifact includes `tf_lite_mtp_drafter`, but implementing the format is not required for native safetensors MTP.

### llama.cpp PR #22673 — MTP for Qwen3.6
- 75% acceptance rate with 3 draft tokens
- >2× speed-up over baseline
- MTP model loads from the same GGUF (not separate)
- Has its own KV cache and context
- Hidden features propagated via "hook" after each ubatch
- Tested on Qwen3.6 27B and 35B-A3B MoE
- `aggregate_accept_rate: 0.8258` in coding benchmark

### Key design decisions from llama.cpp
1. MTP model is a **separate model** but loaded from the **same file**
2. MTP has its **own KV cache** (not shared with main model)
3. Hidden states are extracted via a hook mechanism after each batch
4. Draft tokens verified in a single batched forward pass

## Models with MTP support

| Model | Drafter | Layers | Hidden | Disk |
|---|---|---|---|---|
| Gemma4-E2B | gemma-4-E2B-it-assistant-bf16 | 4 | 256 | 151 MB |
| Gemma4-E4B | gemma-4-E4B-it-assistant-bf16 | 4 | 256 | 159 MB |
| Gemma4-31B | gemma-4-31B-it-assistant-4bit | 4 | 1024 | 283 MB |
| Gemma4-26B MoE | gemma-4-26B-A4B-it-assistant | 4 | 256 | ~200 MB |
| Qwen3.6-35B | built-in (mtp_num_hidden_layers=1) | 1 | shared | in-model |
