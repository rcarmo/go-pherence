# Gemma4 ↔ llama.cpp Alignment Report

Date: 2026-06-18

## 1. Alignment against the 4-dimension rubric

| Dimension | Concrete gate / test | What is compared to llama.cpp / ggml | Tolerance | Status |
|---|---|---|---:|---|
| Output fidelity — MTP strict llama.cpp parity fixture | `make gemma4-mtp-parity GOTMPDIR=$PWD/.gotmp`; strict gate: `GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=... make gemma4-mtp-strict-parity GOTMPDIR=$PWD/.gotmp` | Default fixture checks prompt-context contract, drafter tokens, verifier `[input]+drafted` batch, accepted prefix, bonus/output tokens, and selected logits captured from llama.cpp. Strict real-asset fixture also checks selected verifier logits for the local Gemma4 E4B QAT + BF16 MTP drafter pair. | Token IDs: exact. Selected logits: fixture tolerance; current strict run has stable residual mismatches up to ~0.148. | **Partial**: accepted-token fixture passes; strict selected-logit real-asset parity remains the blocker. |
| C++/ggml structural correspondence — QAT decode + MTP speculative path | `TestGemma4MTPLlamaCPPParityFixture`, `cmd/models/gemma4mtpparity`, `llmgen -mtp-smoke`, `llmgen -mtp-generate -mtp-drafter`; related loader/binding tests for GGUF verifier/drafter tensors | Mirrors llama.cpp graph structure: q-only assistant/drafter with external main-model KV, verifier `[input]+drafted` batch, prompt-context token split, shared target-layer mapping, SWA/full attention windows, per-layer input construction, tied assistant logits before post-projection, and compressed-KV float-shadow verifier staging. | Mostly exact structural invariants; numerical equality checked through fixture gates. | **Partial / structurally aligned**: graph contracts are explicit and guarded; selected-logit parity is not yet exact. |
| SIMD ↔ CPU oracle — GGUF quant kernels | `go test ./loader/gguf -run 'TestDequantRowQ4KToZeroBlock|TestDequantRowQ4KToMatchesGGMLNibbleGroups|TestExpertMatricesQ4KGemvMatchesDequantScalar|TestDequantRowQ8_0ToMatchesScaleTimesInt8|TestQuantizeQ8_0UsesRoundAwayFromZeroWithUnroundedScale|TestDotQ4_0Q8_0MatchesAVX2Reference|TestDotQ4_0Q8_0MatchesScalarReference|TestQuantizeQ8KComputesScaleQuantsAndBlockSums|TestDequantRowQ6KToMatchesScalarReference|TestDotQ6KQ8KMatchesAVX2Reference|TestDotQ6KQ8KMatchesScalarReference' -count=1 -v` | Q4_K, Q6_K, Q8_0, Q8_K packing/dequantization and Q4_0×Q8_0 / Q6_K×Q8_K dot paths are compared against scalar ggml-order or AVX2-order oracles. | Exact integer/block decode where applicable; float dot comparisons use the test-local exact/near-equality thresholds. | **PASS** for the covered quant formats and qdot paths. |
| GPU ↔ CPU oracle — CPU-vs-GPU op trace | `make gemma4-gpu-cpu-parity GOTMPDIR=$PWD/.gotmp`; focused boundary: `TestGemma4MTPVerifierPostAttentionRMSNormGPUParity` | Tagged diagnostic tests compare Gemma4 generation, quantized CPU-vs-GPU layer walks, projection traces, full/early op traces, and the MTP verifier `attn_wo -> attn_post_norm` RMSNorm boundary against CPU/SIMD reference execution. | Op-trace tolerances are test-local and account for GPU reduction-order drift; focused RMSNorm boundary is exact/near-exact per test. | **PASS** for current diagnostic GPU gates; not a proof of strict llama.cpp selected-logit equality. |

## 2. Structural correspondence

The Go MTP path now deliberately follows the llama.cpp/ggml data flow: the prompt fixture contract keeps `[2, 10979]` in prompt KV and uses `236764` as the verifier input token; assistant tied-output logits are taken from the assistant hidden state before `post_projection` produces `h_nextn`; external KV is sourced from llama.cpp-style shared target layers (`n_layer-2` for SWA, `n_layer-1` for full attention); verifier batches are `[input]+drafted` with contiguous positions and mixed sliding/full attention windows; K=V is handled as separate K-norm and V no-scale RMSNorm views over the K projection; verifier attention scale is llama.cpp unit scale; per-layer input rows include `per_layer_token_embd.weight`; and GGUF RoPE, tensor dtypes, scalar norms, `layer_output_scale.weight`, q-only graph metadata, BF16 drafter matrices, and output-head binding are validated at load time.

Numerically, the recent alignment work locked down ggml-style F32 softcap boundaries, fallback RMSNorm double accumulation, iterative RoPE theta progression, BF16 drafter dot reduction order, AVX2 qdot reduction-order regressions, assistant embedding scale `sqrt(n_embd_backbone)`, and GELU/per-layer input ordering. Deliberate divergences remain: Go keeps an explicit typed execution graph and validation layer instead of replaying ggml nodes directly; compressed/TurboQuant MTP verifier staging uses a float-shadow replay bridge; and experimental full-layer verifier batching is still gated rather than default-public.

## 3. Fit / gap analysis

**Fully done:** accepted-token fixture parity for the captured MTP cycle; structural graph validation for drafter/verifier/KV commit; GGUF verifier/drafter metadata and dtype guards; quant SIMD-vs-scalar oracles for the listed Q formats; and diagnostic GPU-vs-CPU op-trace gates.

**Partial:** real-asset end-to-end MTP parity. The runner is valuable and repeatable, and accepted-token parity is green, but the fixture still self-reports `real_asset_acceptance_parity:false` because strict selected-logit parity has six stable mismatches:

```text
row0 token236751 got=14.1096830368042 want=14.126071
row0 token236757 got=15.128488540649414 want=15.1981421
row1 token236751 got=7.351251125335693 want=7.34865713
row1 token236757 got=13.789809226989746 want=13.9382925
row2 token236751 got=20.40302085876465 want=20.2557411
row2 token236757 got=28.634824752807617 want=28.6220856
```

**Not covered / still blocked:** a full real-asset end-to-end acceptance-parity claim across broader prompts, layers, and generation lengths; strict selected-logit equality against llama.cpp for the current captured fixture; and default/public enablement of full-layer verifier batching. Closing the blocker means continuing the direct-port audit at the remaining mismatch path until the verifier selected logits match llama.cpp within the strict fixture tolerance, then expanding fixture coverage before flipping `real_asset_acceptance_parity` and public-readiness flags.
