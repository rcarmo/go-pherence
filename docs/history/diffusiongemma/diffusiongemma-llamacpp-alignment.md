# DiffusionGemma llama.cpp alignment report

_Date: 2026-06-18_

## Headline

The Go DiffusionGemma text/GGUF path now follows the llama.cpp execution-graph shape for the main denoise stack: prompt/canvas embedding semantics, RMSNorm/softcap/tanh-GELU choices, router/top-k weighting, grouped MoE expert execution, dense LM-head/device sampling, and raw-logit self-conditioning have all been moved toward the llama.cpp graph. The required gate `make diffusiongemma-golden-gate` is green locally, including the local real-GGUF smoke and CUDA GPU↔CPU smoke when CUDA is present. The remaining gaps are not hidden: the local fast GGUF smoke is not a full `canvas_length=256` llama.cpp golden, full-canvas end-to-end parity is still represented by per-op/fixture gates plus smoke checks, and vision/multimodal coverage remains partial.

## Four-dimension alignment matrix

| Dimension | Current gate | What it compares | Tolerance / contract | Status |
|---|---|---|---|---|
| Output fidelity | `TestLlamaCppGGUFHi1x1GoldenResponseIDs`, `TestGGUFHi1x1GoTrimmedOutputComparisonGate`, `TestGGUFHi1x1TopLogitProbeGate`, `TestLocalGGUFTinyForwardGoldenTopLogits` via `make diffusiongemma-golden-gate` | Checked-in llama.cpp response/token/logit fixtures and a local real-GGUF smoke. The local smoke currently locks a fast one-row diagnostic top token/logit, not a full 256-canvas graph. | Exact IDs for committed fixtures; local smoke top-1 ID and logit within `1e-3` of its current local reference. | **Partial/pass**: gates are green, but full-canvas real llama.cpp golden is not yet in the fast gate. |
| ggml structural correspondence | `TestDiffusionGemmaTextForwardOpPlanMatchesLlamaCppOrder`, `TestGGUFHiPhaseAlignedTraceStructuralParityGate`, `TestGGUFHiPhaseAlignedLayer0OpsParityGate`, `TestGGUFHiPhaseAlignedInputNormParityGate` | Go op order and phase-aligned row summaries versus llama.cpp: embedding → SC → attention → post-attn → dense MLP + MoE → post-FFN/residual → layer scalar → final norm → LM head. | Exact op order; bounded branch-drift fixture tolerances for committed row-trace summaries. | **Pass/partial**: text graph order is locked; full numerical row-by-row all-layer parity is still not complete. |
| SIMD↔CPU oracle | `TestGGUFQ(4K|8_0|5_0)ExpertRowDotMatchesScalarDequantOracle`, `TestGGUFQ6KLMHeadRowDotMatchesQ8KRoundedDequantOracle`, `TestQ6I8DotAVX2MatchesScalar` | Quantized expert row dots and Q6_K×Q8_K LM-head primitives against scalar dequantized references and AVX2/scalar agreement. | Scalar-oracle tolerances used by the tests; exact ID/shape contracts for malformed input checks. | **Pass** for covered Q4_K/Q5_0/Q8_0/Q6_K CPU/SIMD kernels. |
| GPU↔CPU oracle | `TestLocalGGUFGPUCPUCanvas1Parity` in `make diffusiongemma-golden-gate` under `flock /tmp/go-pherence-gpu.lock` | Local real Q4_K_M GGUF, same prompt/seed/canvas=1, CPU dispatcher versus GPU dispatcher. | Generated token, first argmax, sampled token, accepted count/mask decision must match. Entropy delta is logged, not failed. CUDA hosts fail if CUDA is present but SGEMM/PTX is unavailable. | **Pass/partial**: token/sample/acceptance parity is locked for canvas=1; strict entropy/logit parity and full-canvas GPU↔CPU parity remain gaps. |

## Structural correspondence to llama.cpp

Implemented correspondence points:

- **Text op order:** `BuildForwardOpPlan` matches the llama.cpp `diffusion-gemma.cpp`/`gemma4-common.h` layer sequence: input norm, QKV attention, post-attention norm/residual, dense shared MLP, MoE router/experts, post-FFN norm/residual, region-aware layer scalar, final norm, LM head.
- **Embedding semantics:** prompt embeddings use `embed * sqrt(n_embd)`; canvas embeddings use no-scale RMSNorm semantics after optional self-conditioning.
- **RMSNorm:** epsilon and no-scale variants are represented explicitly; the op-order gate prevents reordering.
- **Activation:** DiffusionGemma MLP/expert activation now uses llama.cpp/ggml tanh-GELU (`LLM_FFN_GELU` / `ggml_gelu`) rather than the removed host “exact-GELU” path.
- **Softcap:** final-logit softcap is implemented as `tanh(x / c) * c`, including a GPU buffer kernel for device sampling paths.
- **MoE routing and experts:** selected experts are grouped once and routed through a `ggml_mul_mat_id`-style backend boundary. Model-level GGUF expert CPU fallback, raw-Q4 alternate residency, sparse LM-head, chunked GGUF LM-head, CPU expert bypass, and partial-layer execution knobs have been removed from the production graph surface.
- **Self-conditioning:** raw-logit self-conditioning with previous-step `temp_inv` is mandatory; the embedding handoff opt-out has been removed. GGUF device LM-head now returns retained device logits state, and `OpSelfCondition` can consume that device state.

Deliberate remaining non-production/reference surfaces:

- Explicit CPU/SIMD dispatcher and CPU expert implementations remain as reference/oracle code paths.
- Trace/fixture tests remain for auditing, not as production graph alternatives.
- The local fast GGUF smoke remains one-row diagnostic because a full `canvas_length=256` CPU unit gate exceeds the normal Go test timeout.

## Fit / gap analysis

### Fully done for the text/GGUF path

- Locked text forward op order against the llama.cpp graph shape.
- Removed known invalid production alternatives: raw Q4 expert mode, host exact-GELU, sparse/top-k LM-head, chunked host GGUF LM-head, raw-SC opt-out, partial-layer graph execution, GPU CPU-expert bypass, and GGUF CPU-prefill env override.
- Covered CPU/SIMD quantized expert and Q6_K LM-head primitive correctness with scalar oracles.
- Covered bounded GPU↔CPU canvas=1 token/sample/acceptance parity on real GGUF.

### Partial / still open

- **Full-canvas llama.cpp golden:** not yet a fast gate. A valid llama.cpp exactness helper requires the model’s `diffusion.canvas_length=256`; the current local unit smoke is intentionally diagnostic and not a full graph golden.
- **Strict entropy/logit parity:** GPU↔CPU smoke still logs entropy drift rather than failing. This must become a stricter sampler/logit oracle after the device `sc_dev` path is fully validated.
- **Full-canvas end-to-end parity:** current confidence comes from per-op fixtures, llama.cpp response fixtures, and bounded smokes, not from a single full-canvas local golden in the required gate.
- **FP8/safetensor path:** several FP8 GPU paths exist, but the main llama.cpp parity target for this report is GGUF Q4_K_M. FP8 is not equivalent coverage for the GGUF backend graph.
- **Vision/multimodal:** operation surfaces exist and some preprocessing/boundary tests pass, but full image-sequence vision reference fixtures are still missing; `ReferenceComplete` remains false for vision tower/insertion.
- **GPU-less hosts:** CUDA-dependent gates skip when no GPU is present, but on CUDA hosts they run/fail as required.

### What closes the gaps

1. Add a committed full-canvas reference artifact generated by llama.cpp’s `llama-diffusion-gemma-eval` for a bounded prompt/canvas and a Go comparison that is suitable for CI time limits, or split it into an explicit slow/reference target outside the fast gate.
2. Promote GPU entropy/logit parity from logged diagnostic to a bounded failing oracle once the device self-conditioning/sampler path is proven against that full-canvas reference.
3. Extend the structural trace fixture beyond the currently locked rows/layers until first-difference analysis is no longer needed for text.
4. Add full multimodal/vision fixtures before marking vision reference-complete.
5. Re-run performance only after the full graph and sampler/logit parity gates are green; performance is not evidence of graph fidelity by itself.
