# Exact F32 attention output interleaving — 1 October 2026

Interleaving four output accumulator chains reduces the attention microbenchmark median by17.1% and fresh request medians by4.6–5.9% relative to the combined exact baseline. All retained hidden values, expanded outputs and word timings stay exact. The new mode is explicit; defaults/services are unchanged. Matching original-engine speed remains unverified.

## Ordered output dataflow

The retained headDim64 shader computes four output columns per lane. Previously it completed all32 key FMAs for column0, then column1, then column2, then column3. The candidate processes key0 for all four columns, then key1 for all four, and so on. Each column still consumes keys0..31 in the same order with the same explicit FMA, probability and value bits.

Tile rescaling still multiplies each accumulator once before that tile's FMAs. Q·K score unroll4, online softmax order, shared Q/K/V layout, barriers, tail masking and final division are unchanged. Independent per-column arithmetic is exposed together to the compiler; no reduction reassociation, precision substitution, activation quantisation or new device feature occurs.

`NewVkAttentionKey32OutputILPF32` requires headDim64 and rejects other dimensions without fallback. `NewVulkanEncoderOriginalQ5AttentionOutputILP` and benchmark arm `vulkan-original-q5-attention-outputilp` explicitly combine it with the retained original Q5 decode4 FFN/F32 arithmetic. The component/combined/default modes remain unchanged. Original source/file checks, packed-only metadata admission, host decoder and word alignment remain intact.

Kernel geometry stays16×16 local size,14,528 shared bytes,4 storage buffers and20 push bytes. Complete encoder native weight/scratch/stage/plan stats match the combined baseline. This mode adds a baseline-compatible shader to the closed regeneration inventory, now31 shaders; no opcode/feature admission is widened.

## Standalone dataflow trials

Intel Iris Xe, same process, current attention-unroll4 baseline,1500 queries/1500 keys/20 heads/headDim64, five alternating-order samples per arm. Every output bit and guard matches baseline on full shape and17-query/37-key tails. First timed samples remain included.

| Trial | Baseline median (ms) | Candidate median (ms) | Change | Decision |
|---|---:|---:|---:|---|
| V tile prefetched into8 registers/lane | 64.985 | 64.807 | −0.27% | No useful gain alone |
| Four output columns interleaved by key | 64.943 | 53.845 | −17.09% | Retain simpler candidate |
| Output interleave with key unroll4 | 64.919 | 53.854 | −17.04% | No added gain |
| V register prefetch + output interleave | 64.920 | 53.284 | −17.92% | Diagnostic only; no trained acceptance |

V-register prefetch combined with interleave was slightly faster in this short diagnostic, but retains extra live registers and adds another scheduling change. It was not model-qualified and is not in runtime. Only output interleave is retained. Temporary prefetched/unrolled shaders and harness remain in the evidence directory, not production packages.

## Fresh trained five-repeat requests

Baseline `vulkan-original-q5-exact-combined`, candidate outputILP, pinned original turbo Q5 stored values, four CPU threads, greedy language-specific decode and F32 flash attention. Real Silero VAD, preserved audio gaps and CPU word alignment are enabled in the indicated arms. Each fresh baseline process precedes candidate; order is not randomised. Five requests per arm include the first, with no discarded samples.

| Fixture | Combined baseline median (s) | OutputILP median (s) | Change | All returned outputs |
|---|---:|---:|---:|---|
| JFK English | 6.694 | 6.301 | −5.87% | Exact |
| MINDS Portuguese row0 | 6.664 | 6.297 | −5.50% | Exact |
| MINDS French row0 | 6.138 | 5.780 | −5.84% | Exact |
| JFK native VAD + words | 7.895 | 7.515 | −4.82% | Exact speech/audio windows, segments and word times |
| Two speech groups, VAD + words | 15.851 | 15.124 | −4.58% | Exact speech/audio windows, segments and word times |

Model/input pins, language, VAD/words/gap flags, CPU budget and encoder stats match within every pair. Preparation/load/full-arm totals stay separate: JFK five-request full-arm37.394→35.356s, Portuguese37.144→35.340, French34.557→32.740, JFK VADwords43.308→41.513 and groups83.250→79.572. These are not original-engine cold workflow comparisons, and the gain is not added to historical percentages.

## Expanded and independent gates

Five candidate requests each on extra MINDS PT rows1/2 (including43.76s), podcast20s no-VAD, and podcast20s native-VAD/words match every retained combined-baseline output exactly. These baseline samples come from the earlier combined window, so this is an output gate, not fresh speed acceptance. No FC1 quantisation word/segment change or20ms alignment shift appears. Five silence/VAD/word requests return no windows; median0.01975s.

Pinned complete encoder synthetic mel yields1,920,000 bit-identical hidden values. Three cancellation checkpoints, native drain where needed, fresh bitwise reuse and complete native cleanup pass; latest run records1241 checkpoints. Native model stats stay unchanged.

Independent full-matrix float64 attention reference passes six shapes ×three input signs, including4096-sequence bounds,32 heads, and tails. Existing2e-5 absolute/relative analytic budget is unchanged; exact native-baseline/hidden comparisons are separate. Model-free reflection/constructor tests cover ABI, cancelled construction, required head dimension, device shared-memory bounds and unchanged defaults.

This is exact retained-Go parity, not independent acoustic correctness. Earlier Go-vs-original timing gaps and human-label disagreements remain unchanged. No original engine run occurred in this window.

## Measured retained profile

The profiler explicitly admits outputILP, loads the complete verified original source, fences stages individually for three passes, and requires final hidden equality with whole-plan execution. No activation dump is enabled. These synchronisation-heavy host timings identify costs, not request speed acceptance:

| Operator group | Mean fenced time per encoder pass (s) |
|---|---:|
| Attention outputILP | 1.729 |
| Packed Q5 decode4 FC1 | 0.994 |
| Packed Q5 decode4 FC2 | 1.032 |
| Q/K/V/O combined | 1.191 |
| Convolutions | 0.118 |
| GELU | 0.076 |
| Add | 0.074 |
| Normalisation | 0.064 |

The prior combined attention profile was2.094s; this separate-profile attribution is consistent with the microbenchmark but is not a paired request baseline. FFN and Q/K/V/O remain substantial costs. No new projection arithmetic was introduced.

## Verification and isolation

`make model-layout-check host-build host-vet host-test docs-check`, whole-tree race tests, independent marked ARM64/RISC-V builds, latest affected race/vet/tests pass. All31 stored/embedded shaders validate and regenerate with normalised identity. Checker tests plus optional integer-dot checker tests pass (7 tests/90 assertions). Final native independent and complete hidden/cancel gates executed and passed on the retained tree. The read-only independent review timed out after60s; no review approval is claimed.

The runner retained CPU4/8GiB/no-swap, heap target4GiB, physical Intel render node, no network/read-only root, Qwen idle and host available memory≥6GiB guards. Qwen LAN/Gemma/defaults/services/resources were untouched. Native test deadlines stayed≤120s. Outer240s tool timeouts were followed by continued guards and actual container exit0/noOOM inspection/removal; timeout itself was not treated as drain.

[Hashed evidence](../../benchmarks/speech-foundations/vulkan-attention-outputilp-20261001/) retains raw timings, trained/expanded JSON, settings, synthetic diagnostic source/SPIR-V/harness, native states, independent reference/hidden tests, full build/race/cross/shader logs, profile, comparison script, runner and provenance. No unqualified prefetch variant remains in runtime.

## Remaining goal

Continue measured exact-F32 FFN/Q/K/V/O/attention work without broadening failed quantisation. The current gains preserve Go outputs but do not resolve original timing/quality gaps. Matched original Vulkan+flash+VAD latency, independent acoustic/word timing, long-form/resume and fault coverage remain open. This checkpoint is not a default change or deployment; the original-speed objective remains unmet.
