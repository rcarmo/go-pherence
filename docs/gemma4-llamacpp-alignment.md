# Gemma4 llama.cpp alignment report

_Last updated: 2026-06-18._

This is a fit/gap summary for the Gemma4 E4B QAT GGUF verifier plus BF16 MTP drafter path. The implementation goal is a direct port of the llama.cpp Gemma4/MTP backend graph, with no deliberate semantic deviations. The default accepted-token gate is green, but the strict selected-logit fixture is still the active blocker.

## 1. Rubric alignment

| Dimension | Gate / test | What it compares | Tolerance | Status |
| --- | --- | --- | --- | --- |
| Output fidelity: MTP strict llama.cpp parity fixture | `make gemma4-mtp-parity GOTMPDIR=$PWD/.gotmp`; strict: `make gemma4-mtp-strict-parity` with `GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=tmp/gemma4-mtp-llamacpp-fixture.json`, `GO_PHERENCE_GEMMA4_MAIN=models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf`, and `GO_PHERENCE_GEMMA4_MTP_DRAFTER=models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf` | Default gate checks committed llama.cpp token/acceptance contract plus real-asset graph cycle when assets exist. Strict gate compares selected verifier logits from local `llama.cpp --flash-attn on` for prompt KV `[2,10979]`, input `236764`, draft `[564,236789]`, verifier batch `[236764,564,236789]`, accepted prefix `2`, bonus `236757`. | Strict fixture `logit_tolerance=0.001`; accepted-token parity exact. | **Partial / gap**: default accepted-token/bonus parity is green; strict selected verifier logits are red with five residual mismatches after removing the Gemma4 layer-output BF16 graph deviation and enabling the ggml-style F16 flash path by default. `RealAssetAcceptanceParity=false` is correct. |
| ggml structural correspondence: QAT decode + MTP speculative graph | Focused model tests around `TestGemma4MTPLlamaCPPParityFixture`, `TestAssistantLogitsIntoDoesNotApplyVerifierSoftcapOrSuppressBias`, `TestRunMTPDrafterStepUsesAssistantHiddenForLogitsAndPostProjectionForHandoff`, `TestRunMTPDrafterStepScalesBackboneEmbeddingByBackboneWidth`, `TestMapGemma4MTPDrafterKVSourceLayersUsesLlamaCppSharedTargets`, `TestMTPVerifierAttentionPlanMixedSlidingAndFullRanges`, `TestForwardMTPPromptLayerGemma4AttentionUsesUnitScale`, `TestProjectMTPVerifierLayerQKVBatchGemma4KEqV` | Direct structural parity with llama.cpp/ggml graph pieces: assistant BF16 drafter data flow, verifier `[input]+drafted` batch, PLI order, external KV source mapping, K=V handling, unit attention scale, mixed SWA/full windows, logits-vs-handoff split. | Mostly exact structural invariants and row-wise oracle equality in unit tests; selected-logit tolerance only in strict fixture. | **PASS for covered structure; partial for end-to-end numeric graph**. The explicit graph shape is represented and guarded, but strict final selected logits still drift. |
| SIMD ↔ CPU oracle: Q4_K/Q6_K/Q8_0/Q8_K and GGUF qdots | `make gemma4-mtp-parity` includes `go test ./loader/gguf -run '...TestDotQ4_0Q8_0MatchesAVX2Reference...TestDotQ6KQ8KMatchesAVX2Reference...TestDotQ6KQ8KMatchesScalarReference...'`; related coverage includes Q4_K dequant/GEMV, Q8_0 quant/dequant, Q8_K scale/sums. | Quant decode and dot-product helpers used by GGUF/QAT projection paths are checked against scalar/dequantized or AVX2-order references matching ggml block layouts and rounding expectations. | Test-specific exact/near-exact numeric equality; failures are hard assertions. | **PASS for covered quant primitives**. This does not by itself prove whole-model selected-logit parity. |
| GPU ↔ CPU oracle: CPU-vs-GPU op trace | `make gemma4-gpu-cpu-parity GOTMPDIR=$PWD/.gotmp` runs diagnostic, GPU-locked tests including `TestGemma4GPUGenerate`, `TestGemma4QuantizedCPUvsGPULayerWalk`, `TestGemma4CPUvsGPUProjectionTrace`, and `TestGemma4QuantizedCPUvsGPUOpTrace`. | CPU and CUDA/GPU execution paths for Gemma4 generation/projection/op-trace surfaces under bounded KV memory. This is an implementation-oracle gate, not a llama.cpp selected-logit fixture. | Test-specific trace tolerances/hard assertions. | **Covered by required gate, not re-run for this report**. It validates CPU↔GPU consistency, but the current strict blocker is CPU/ggml-vs-llama selected logits. |

## 2. Structural correspondence to llama.cpp/ggml

The current implementation mirrors the llama.cpp graph in the following places:

- **MTP execution graph**: `MTPExecutionGraph` represents hidden-state-conditioned drafter steps, verifier `[input]+drafted` batch, contiguous verifier positions, and acceptance-dependent KV keep window (`accepted_prefix + bonus`). Commit helpers recompute acceptance from verifier logits before mutating float or compressed KV.
- **BF16 MTP drafter**: GGUF BF16 assistant matrices are required and executed through BF16 rows; the drafter can run without F32 matrix shadows. Assistant logits come from the assistant tied `token_embd.weight`, while `post_projection` only produces `h_nextn` for handoff, matching llama.cpp.
- **RoPE and positions**: Gemma4-local llama.cpp-style RoPE progression and GGUF `rope_freqs.weight` validation are in the Gemma4/MTP path. Prompt fixture contract is `[10979,236764]` -> prompt KV `[2,10979]` plus input token `236764`.
- **Embedding scale**: assistant pre-projection uses token embedding scaling by `sqrt(n_embd_backbone)`, matching the llama.cpp assistant graph.
- **Verifier QKV norms and K=V**: Q/K norm validation follows the sequential llama.cpp-style path. When V projection is omitted/shared, V is derived from the original K projection with Gemma4 no-scale RMSNorm while K receives K-norm; this is locked for verifier batch projection and ordinary generation paths.
- **KV sharing and attention windows**: Gemma4 assistant external KV source mapping follows llama.cpp shared targets (`n_layer-2` for SWA, `n_layer-1` for full attention), including duplicate source layers. Mixed sliding/full verifier ranges are locked for `[input]+drafted` batches.
- **Attention scale**: Gemma4 verifier attention uses llama.cpp's unit scale (`f_attention_scale = 1.0f`), not `1/sqrt(head_dim)`.
- **PLI and GELU/FFN order**: per-layer input construction follows `norm(project(hidden) / sqrt(hidden)) + per_layer_token_embd[token] * sqrt(hidden_per_layer)`, then `1/sqrt(2)`. The prompt/verifier layer path applies PLI, attention residual/post-norm, FFN norm/GELU-gated MLP/post-norm, residual, and layer output scale in llama.cpp order for the covered dense/QAT path. The previous Gemma4 layer-output BF16 store has been removed from the production Gemma4 path because llama.cpp feeds `out_scaled`/`l_out` directly to the next layer without a BF16 cast.
- **Verifier tail/logits**: dense batched decode tail applies final norm and GGUF LM-head paths correctly, including separate `output.weight` when present and verifier-only softcap/suppress-token behavior; assistant logits explicitly do not inherit verifier final-logit softcap or suppress bias.

There are no intentional semantic divergences in the Gemma4 E4B QAT GGUF + BF16 MTP path. Remaining non-finalized surfaces are implementation staging choices: the production verifier full-layer path still lowers nonzero verifier batch rows through the sequential row/layer oracle by default, while the full-layer batch scaffold is gated behind `GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS`; public/default MTP generation remains gated until strict real-asset parity is solved.

## 3. Fit / gap analysis

**Done / fit:**

- Default MTP accepted-token and bonus-token parity is green via `make gemma4-mtp-parity GOTMPDIR=$PWD/.gotmp`.
- The default fixture remains trimmed and committed, so ordinary CI locks token/acceptance semantics without requiring local selected-logit probes.
- The strict fixture path exists and fails loudly when the fixture env is missing.
- Loader and graph contracts validate GGUF verifier/drafter tensor presence, dtypes, shapes, attention/RoPE metadata, BF16 drafter matrices, layer scales, PLI tensors, and compressed-KV staging consistency.
- Quant primitive gates cover the QAT/GGUF dot/dequant surfaces used by the verifier path.

**Partial:**

- ggml graph structure is represented and guarded, but the strict selected-logit comparison is not yet green.
- GPU↔CPU consistency is represented by a dedicated diagnostic gate, but it is orthogonal to the active llama.cpp selected-logit blocker.
- Full-layer verifier batch lowering exists as a gated scaffold with synthetic parity; it is not default-enabled and is not the basis for the current public-readiness claim.

**Not covered / active blocker:**

- The strict local `llama.cpp --flash-attn on` selected verifier logits still mismatch at tolerance `0.001`. Live strict run after removing the Gemma4 layer-output BF16 graph deviation and enabling the ggml-style F16 flash path by default:

```text
row0 token236751 got=13.99431324005127  want=14.126071   delta=-0.131758
row0 token236757 got=15.062721252441406 want=15.1981421 delta=-0.135421
row1 token236751 got= 7.268852233886719 want= 7.34865713 delta=-0.079805
row1 token236757 got=13.671273231506348 want=13.9382925 delta=-0.267019
row2 token236751 got=20.173765182495117 want=20.2557411 delta=-0.081976
row2 token236757 got=28.622596740722656 want=28.6220856 delta=+0.000511 (within tolerance)
```

The fixture still reports `real_asset_acceptance_parity:false`, and that is intentional until this strict selected-logit gate is green. Closing the gap means finding the remaining exact llama.cpp/ggml numeric semantic difference in the verifier graph and making `make gemma4-mtp-strict-parity GOTMPDIR=$PWD/.gotmp` pass against the local fixture, without relaxing tolerance or changing the fixture. Only then should `RealAssetAcceptanceParity`, public/default generation readiness, or full-layer verifier batch default enablement be promoted.
