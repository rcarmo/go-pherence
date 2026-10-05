# Original Vulkan graph attribution — 1 October 2026

The original Q5 backend's integer-dot path is the largest identified compute difference. Three diagnostic runs put original encoder projections at about 1.025 s with integer-dot enabled and 3.852 s disabled. The current Go projections total 3.371 s in separately fenced stage measurements. These timings explain a priority; they do not establish precision-equivalent speed acceptance.

## Diagnostic boundaries and pins

Original source: whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`. Go starting revision: `c0eaa55c862502681cea54d092174d3f7178c7cd`. Turbo original Q5_0 model pin: `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`; original F16 model pin: `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`. JFK WAV pin: `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`. Existing binary/library hashes, commands and runtime image are retained with evidence.

Original runs enabled the existing `GGML_VK_PERF_LOGGER` timestamp profiler at frequency 1. It synchronises around operations and changes execution cost. Original operator figures below are device timestamps; `encode` is original host phase time and additionally includes convolution/cross-graph execution and setup/conversions. Go stage attribution individually fences each stage and reports host submission/fence time. Profiling times are not replacements for uninstrumented process/request baselines; the two instrumentation boundaries are not identical.

Both use Intel Iris Xe, four CPU threads, fixed English/greedy settings and flash attention. Q5/default, Q5/no-integer-dot and Q5/no-F16 arms have three separate profiler processes. F16 and the combined no-integer-dot/no-F16 arm have one diagnostic run each. No original source or production binary was edited.

## Original encoder measurements

The encoder profiler block includes 128 square Q/K/V/O matmuls, 32 FC1 matmuls, 32 FC2 matmuls and 32 flash-attention calls. Projection totals aggregate all 192 matmuls; attention totals aggregate all 32 calls. Listed values are medians across the available runs.

| Original diagnostic arm | Runs | Encoder projections (s) | Flash attention (s) | Host encode phase (s) |
|---|---:|---:|---:|---:|
| Q5, default integer-dot/F16 capability | 3 | 1.025 | 1.580 | 3.232 |
| Q5, integer-dot disabled | 3 | 3.852 | 1.543 | 6.113 |
| Q5, F16 capability disabled | 3 | 1.017 | 1.454 | 2.992 |
| Q5, integer-dot and F16 disabled | 1 | 2.424 | 1.435 | 4.414 |
| Original F16 weights | 1 | 3.556 | 1.586 | 5.876 |

Disabling integer-dot adds about 2.83 s to encoder projection timestamps. Disabling F16 does not remove the integer-dot advantage; it also changes attention and other precision-dependent choices. The faster no-F16 arm is an ablation, not a validated production setting or general accuracy result. Combined disable changes fallback kernels and conversion costs; it is not an isolated F32 arithmetic oracle.

All original diagnostic runs produced identical JFK text, content token IDs and segment offsets (one segment, 0–10400 ms). Token probabilities differ and were not asserted equal. These short output checks do not qualify word timing, multilingual robustness, VAD, long-form accuracy or an approximate arithmetic mode.

## Current Go attribution on the same original Q5 values

The diagnostic stage harness now explicitly admits `vulkan-original-q5-mlp`, uses the verified original file for FFN packed preparation, and records output/input names with backend selection. Full metadata, geometry, pinned input and physical-device gates remain. The existing default and score-ILP modes stay unchanged; unknown modes fail before loading. Each individually fenced stage runs three times and the final hidden output must equal the whole encoder graph.

| Go operator group | Mean per encoder pass (s) |
|---|---:|
| Q projection | 0.301 |
| K projection | 0.300 |
| V projection | 0.300 |
| O projection | 0.296 |
| Packed-Q5 FC1, F32 activation/accumulation | 1.064 |
| Packed-Q5 FC2, F32 activation/accumulation | 1.108 |
| All projections | **3.371** |
| Key32 score-ILP F32 attention | **2.162** |
| Convolutions | 0.116 |
| GELU | 0.077 |
| Add | 0.077 |
| Normalisation | 0.066 |

An earlier timeline estimate said Go projections were 3.67 s; the retained stage samples sum to 3.371 s. Original and Go phase totals also differ in cross-attention preparation placement: the original `whisper_encode_internal` computes a separate cross graph after the encoder, while Go decoder setup belongs to its request flow. These attribution figures must not be relabelled a matched whole-workflow benchmark.

## Precision and arithmetic path

Source inspection at the pinned original revision establishes these conditions:

- `ggml_vk_mul_mat_q_f16` (`ggml-vulkan.cpp`, around line 9325) selects Q8_1 RHS quantisation when integer-dot is available, F32 RHS is contiguous and the shape passes its packing conditions. It falls back if the Q8_1 pipeline is unavailable.
- `quantize_q8_1.comp` computes a block absolute maximum, scale `amax/127`, rounded signed bytes and sum correction. Scale and scaled sum are stored as two F16 values in the packed block. This rounds activations and scales; Go's retained path keeps activations F32.
- `mul_mmq.comp` uses `GL_EXT_integer_dot_product` and packed Q8_1 RHS. In `mul_mmq_funcs.glsl`, Q5_0 reconstructs unsigned five-bit weights and accumulates packed integer dots, then evaluates `A.d * (q_sum * B.d - 16 * B.s)`. This differs from ordered dequantised-F32 element FMAs.
- Shader generation chooses FP16 or FP32 accumulator definitions according to capability/variant. `GGML_VK_DISABLE_F16` affects several generated pipeline choices; the device log and ablation alone do not identify every per-operation accumulator variant.
- Whisper's flash-attention branch copies K/V into cache storage; the context defaults its intermediate/cache type to F16. Vulkan attention selects F32 accumulation when F16 capability is unavailable, explicit operation precision requests F32, or K is BF16. Go attention instead keeps Q/K/V and its ordered online reductions F32.

The original log names `MUL_MAT q5_0`, not every selected shader variant. Enabled/disabled device feature logs, source conditions and repeatable timing ablations strongly identify the integer-dot path's contribution; a per-dispatch shader/operand dump was not collected. No precision substitution or capability admission was added to Go in this window.

## Next implementation contract

The next speed candidate should be a separately named native **Q5_0 × Q8_1 projection**, retaining the exact original block/scaling semantics and requesting integer-dot support explicitly. Start with an independent scalar oracle for Q8_1 rounding, F16 scale/sum correction, integer dot and block accumulation. Cover ties, negative/zero/subnormal scales, saturation, block/tail geometry, cancellation and output drift. Hardware admission must negotiate only required features and reject unsupported devices without an F32/CPU fallback.

Then compare analytic and pinned model tensors, whole encoder hidden drift, text/tokens/segments, native VAD/word timing and matched end-to-end boundaries. Quantised activation acceptance cannot reuse the existing bitwise-F32 promise; measured error and explicit mode approval are required before broader use. This is an implementation priority, not completed code or an accepted accuracy trade-off. Attention precision/layout differences are a secondary measured target.

## Verification, isolation and evidence

Only the diagnostic Go stage harness changed. Admission tests repeated ten times; `make model-layout-check host-build host-vet host-test docs-check`, affected Whisper race tests and final affected vet/tests passed. No kernel/default or original source changed. No foreign-build rerun is claimed for this test-only change.

The Intel runner retained four-CPU quota, `taskset -c 0-7`, 8 GiB memory/memory-plus-swap cap, Go heap target 4 GiB, no network and read-only root. Idle Qwen `/slots` and host available memory ≥6 GiB guards stayed active. Final gated container exited 0 without OOM or guard abort. At 09:00 UTC, native/build/candidate work drained, only pre-existing `wrdp` containers remained, no Go process was active and Qwen slot0 was idle. `@llama` received explicit release. No independent review was completed.

[Evidence directory](../../benchmarks/speech-foundations/whisper-original-attribution-20261001/) retains compressed exact original profiler logs, full original JSON outputs, per-stage Go samples, feature-ablation environments, runner, source-aware analysis script, attribution totals, container states, provenance and checksums. The script parses `MUL_MAT` rows with `n=1500` and the 32-call flash encoder block; initial parsing errors were corrected before publishing totals. The report aggregates phases without changing original samples. Reproduce with new output paths and profiler flags; do not overwrite retained baselines.

The original-engine speed objective and matched Vulkan+flash+VAD/independent long-form/resume/accuracy gates remain open.
