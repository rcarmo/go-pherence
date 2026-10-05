# Experimental Whisper integer-dot FFN — 1 October 2026

The explicit Q5_0×Q8_1 FFN candidate improves request medians by7.9–10.9% on the five retained fixtures. Portuguese text and content tokens remain exact, but its segment end changes from7.36s to9.04s; the pinned original engine returns7.36s. **Timing acceptance fails.** The candidate stays explicitly experimental, with no default or service changes. It does not meet the original-speed goal.

## Arithmetic and implementation

`NewVulkanEncoderOriginalQ5IntegerDotMLP` and the benchmark-only `vulkan-original-q5-integer-dot` arm require explicitly enabled integer-dot and pinned original Q5 FFN weights. Encoder FC1/FC2 use resident packed weights, Q8_1 activation scratch and tiled integer-dot. Other weights, attention and decoder retain the existing F32 arithmetic; flash attention remains the score-ILP key32 path. No CPU fallback or device-feature upgrade occurs.

`VkLinearQ5IntegerDotSet` streams original weights and returns quantise/project stages over owned immutable weight views. Both kernels and storage follow the existing lane/plan/fence lifetime. Admission checks geometry, rank, scratch size, buffer/device ownership and aliasing. In-flight close refuses cleanup; failed construction preserves unresolved owners for retry. Model-free tests cover stage bindings, aliases, cancellation, closed/disabled owners, reader/extent failure and quantiser creation rollback.

Quantisation uses32-value blocks, F32 maximum, `d=amax*(1/127)`, `inv=127/amax`, signed-byte rounding and F16 scale/scaled sum. Projection reconstructs unsigned Q5 bytes and computes `dW*(integerSum*dQ-16*sQ)`, with F32 correction and explicit FMA across blocks. This is not ordered per-element F32 arithmetic or a claim of original MMQ accumulation equivalence.

Activations must be finite with magnitude≤1000; nonzero block maxima must be≥1e-30. The shader emits NaN scale/sum markers for invalid blocks instead of performing unsafe float-to-integer conversion. Encoder output admission rejects resulting nonfinite values. Native tests cover NaN, positive/negative infinity, over-bound magnitudes and tiny blocks. Existing F32 kernels/contracts/defaults are unchanged.

Weight bytes stay1,184,890,880; scratch increases93,696,000→104,496,000 bytes (+10.8MB). Plans stay34; stages390→454. Packed-only benchmark loading removes the CPU encoder and forbids CPU fallback. Public widened-source constructor verifies original packed values against the supplied source.

## Independent original quantiser comparison

Original revision: whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`; model SHA256 `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`. The comparison extracts existing generated ×4 subgroup/non-subgroup SPIR-V, validates it, and executes it in a standalone C Vulkan harness. It enables only the original quantiser's required Int8/storage16/subgroup features in that separate diagnostic device; Go production admission is not widened to accept the original modules.

Actual JFK layer0 FC1/FC2 inputs were captured from the retained F32 encoder. Diagnostic downloads happen outside timed inference and cannot be interpreted as speed measurements. Input pins:

- FC1: `e43e03363bafbf2f3cdbca5290e92cd5547406c3a601e37f80d365926a8333db` (1,920,000 F32 values).
- FC2: `07cdaf0e8c75e8fa7c1977c6a0bbf2a753fb7b27f3402ba575eaf9dbe0113f3f` (7,680,000 F32 values).

The original ×4 layout is unpacked independently into normal blocks. Both original variants match every quantised byte, scale and scaled sum:60,000 FC1 blocks +240,000 FC2 blocks. The earlier inverse `1/d` differed in2 and4 quantised words/sums respectively. A direct division trial still differed; matching the compiled original's `127/amax` resolved all differences. Original source had been inspected inconsistently at first; pinned `git show` plus disassembly established the actual two-word push constants and flat dispatch. The failed first C-harness run populated only40/160 blocks due to wrong dispatch/push ABI; its output was rejected and regenerated. Failures are retained, not accepted as oracle results.

GLSL Round tie direction remains implementation-defined. Sigma's analytic nearest-even tie tests pass; this300,000-block agreement qualifies these inputs/device, not every layer, exact tie, platform or complete original projection. Independent scalar Q8/Q5 block tests205×5 still pass after the correction.

## Trained FFN operator diagnostics

Pinned layer0 original weights and the captured inputs, no bias, five alternating-order baseline/candidate pairs. Each candidate sample includes quantisation plus projection; construction and host loading are excluded. Six selected edge/interior outputs per shape match the independent original-byte block/FMA oracle. Full output scalar parity is not claimed.

| Layer0 tensor | Grouped Q5/F32 median (ms) | Integer-dot + quantise median (ms) | Change | Maximum absolute F32 drift |
|---|---:|---:|---:|---:|
| FC1 | 32.877 | 22.718 | −30.90% | 0.008897 |
| FC2 | 33.841 | 23.971 | −29.16% | 0.029331 |

Latest pinned rerun also passes arithmetic/guard/cleanup checks. These are operator results, not acoustic accuracy acceptance.

## Five-repeat request comparison

Matched original stored values, physical Intel Iris Xe, four CPU threads, greedy language-specific decode, F32 key32 flash attention. Baseline is `vulkan-original-q5-source`; candidate is the explicit integer-dot FFN arm. Each process loads/prepares once and reports five requests; all five retained samples contribute to medians, including the first. Baseline fixture process precedes candidate; order is not randomised. Final bounded-quantiser candidate runs are reported below, not earlier unrestricted samples.

| Fixture | Baseline request median (s) | Candidate median (s) | Change | Returned output |
|---|---:|---:|---:|---|
| JFK English | 6.935 | 6.223 | −10.27% | Exact windows/text/content tokens/segment times |
| MINDS Portuguese | 6.920 | 6.229 | −9.97% | Text/content tokens exact; segment end7.36→9.04s **fails** |
| MINDS French | 6.341 | 5.649 | −10.92% | Exact windows/text/content tokens/segment times |
| JFK native Silero VAD + words | 8.096 | 7.459 | −7.87% | Exact speech/audio windows, segments and word times |
| Two speech groups, VAD + words | 16.360 | 14.958 | −8.57% | Exact windows, segments and word times |

The Portuguese mismatch occurs consistently across all five samples. A separate original-engine Q5/Vulkan/flash diagnostic returned the baseline7.36s endpoint and identical text. No output clamping, decoder heuristic, fallback, tolerance widening or timestamp special case was introduced. Content token IDs do not include every decoder timestamp decision; they cannot establish timing parity.

Five silence/VAD/word requests return no windows; median0.01960s. This checks no hallucinated speech in that fixture, not general silence quality.

Preparation and full-arm timings remain separately retained. JFK full-arm38.518→35.109s; Portuguese38.421→35.092; French35.616→32.195; JFK VADwords44.353→41.025; groups85.684→78.498. These five-request arms do not establish single cold original-engine workflow parity. Historical original Q5 request timings use different precision/boundaries; no overall original speed win is claimed.

## Hidden drift, cancellation and resources

A synthetic mel pass over the pinned complete encoder changes1,919,998 of1,920,000 hidden values; maximum absolute drift7.2178, mean0.04863. These are measured drift, not a passed F32 equality/quality threshold. Three checkpoint cancellation points, drain where needed, fresh reuse and bitwise candidate-repeat identity pass; latest run records1304 checkpoints. Both candidate/native allocations return to their baseline after close.

Isolated runner retained CPU4/8GiB memory/no swap, Go heap target4GiB, physical Intel render node, no network and read-only root. Qwen idle and host available memory≥6GiB guards remained active. Qwen LAN/Gemma/service defaults were untouched. Native test deadlines were≤120s. Interrupted/outer-timeout containers were checked until actual exit0/noOOM and removed before reuse/release.

## Verification and retained evidence

Affected tests/vet pass, as do `make model-layout-check host-build host-vet host-test`, whole-tree race tests and independently marked ARM64/RISC-V builds. Latest affected race tests pass. The existing28-shader gate passes; the two new optional embedded shaders pass modern-shaderc regeneration, strip, validator and fixture-equality checks through `scripts/check-vulkan-integer-dot.ts`. Its failure/overwrite unit test passes. No independent review approval was obtained; the delegated C-harness attempt timed out and produced no runner.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-integer-dot-ffn-20261001/) retains compressed actual F32 inputs/original quantised outputs, original SPIR-V and C source, exact benchmark JSON, failed runs, native/container states, commands, runner, phase-gate logs, provenance and comparison script. The large fixture outputs are compressed without changing their contents. [Optional shader fixtures](../../backends/vulkan/testdata/integer-dot/README.md) document regeneration and numerical limits.

## Next work

The candidate is a useful speed experiment but fails preserved timing. Diagnose original accumulation/precision and the timestamp-logit boundary before considering broader arithmetic. Do not mask the endpoint change. Q/K/V/O projections and attention remain F32 bottlenecks; any quantised expansion requires a separately named candidate and new model/timing gates. Independent acoustic accuracy, long-form/resume and fault coverage remain open. The original Vulkan+flash+VAD speed objective remains unmet.
