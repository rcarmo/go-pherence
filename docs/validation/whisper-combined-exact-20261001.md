# Combined exact Q5 decode4 and F32 attention unroll — 1 October 2026

The explicitly combined candidate reduces fresh Go request medians by2.5–3.8%, with every retained output exact. The gain was measured on the combined graph, not calculated by adding component gains. Hidden values, resource geometry and expanded word timings remain unchanged. This is not original-engine speed acceptance; defaults and services are unchanged.

## Implementation

`NewVulkanEncoderOriginalQ5ExactCombined` and benchmark arm `vulkan-original-q5-exact-combined` select the two previously qualified kernels together:

- Original packed Q5 FC1/FC2 use the four-lane decoder; dequantised values and ordered F32 projection arithmetic stay unchanged.
- F32 key32 score-ILP attention uses the four-channel Q·K unroll, preserving each score's channel/FMA order and the existing softmax/output recurrence.

Attention requires headDim64; all other dimensions are refused without a kernel/device/CPU fallback. Source/file value and packed-only metadata admission are unchanged. The benchmark loads the host decoder without CPU FFN widening, then clears the CPU encoder after successful native construction. No activation quantisation or device feature negotiation is introduced.

The two component-only modes remain separate. Ordinary encoder/default/kernel choices are unchanged. This combined mode adds no shader or opcode admission. Existing30-shader inventory remains intact.

## Fresh five-repeat requests

Intel Iris Xe, four CPU threads, pinned original turbo Q5 values, language-specific greedy decoding, retained F32 flash attention, real Silero VAD and CPU word alignment when enabled. Baseline is `vulkan-original-q5-source`; each fresh baseline process precedes the combined process. Each arm includes five requests, including the first; no samples are discarded. Order is not randomised, and small gains are device/workload-specific.

| Fixture | Baseline median (s) | Combined median (s) | Change | All returned outputs |
|---|---:|---:|---:|---|
| JFK English | 6.909 | 6.689 | −3.18% | Exact |
| MINDS Portuguese row0 | 6.880 | 6.663 | −3.16% | Exact |
| MINDS French row0 | 6.374 | 6.129 | −3.84% | Exact |
| JFK native VAD + words | 8.139 | 7.875 | −3.25% | Exact speech/audio windows, segments and word times |
| Two speech groups, VAD + words | 16.303 | 15.903 | −2.46% | Exact speech/audio windows, segments and word times |

Model/input hashes, language, VAD/word/gap-preservation flags, CPU budget and encoder stats agree within every pair. Loading, preparation and complete five-request arm time are recorded separately. Full-arm totals improve38.657→37.294s for JFK,38.255→37.229 for Portuguese,35.631→34.492 for French,44.417→43.312 for JFK VADwords and85.325→83.271 for speech groups.

These five-request arms are not single cold original-engine workflows. No original-engine execution was collected in this window; earlier original precision/timing-boundary differences must not be treated as resolved. Component-only gains are documented in [attention unroll](vulkan-attention-unroll4-20261001.md) and [Q5 decode4](vulkan-q5-decode4-20261001.md), not summed here.

## Expanded and hidden gates

Five combined requests per additional recording match every retained baseline output: MINDS PT rows1/2 (including43.76s), podcast20s without VAD, and podcast20s with native VAD/word alignment. Their baseline samples come from the earlier [expanded quality window](whisper-fc1-expanded-quality-20261001.md), so this is an output gate, not a fresh paired speed claim. The quantised FC1 word change and20ms alignment shift do not occur in this exact candidate.

Five silence/VAD/word requests return no windows; median0.02022s. A pinned complete encoder on synthetic mel produces1,920,000 bit-identical hidden values. Memory/stage/plan stats match baseline. Three cancellation checkpoints, native drain if needed, candidate-bitwise reuse and complete allocation cleanup pass; final test records1274 checkpoints.

Model-free tests cover nil/cancelled/invalid-source admission, benchmark/stage-mode admission and unchanged defaults. The fixed-dimension refusal and kernel/stream lifetime checks remain covered by the existing component tests. Exact Go parity is not independent acoustic correctness: pre-existing Go-vs-original timing gaps and supplied-label disagreements remain unchanged.

## Combined graph attribution

Stage profiling explicitly admits the combined mode. It loads the complete verified original source and separately fences each stage for three passes. Final hidden output must match whole-plan execution exactly. This profile enables no activation dump and is not used for request speed acceptance.

| Operator group | Mean fenced time per encoder pass (s) |
|---|---:|
| F32 key32 attention unroll4 | 2.094 |
| Packed Q5 decode4 FC1 | 0.993 |
| Packed Q5 decode4 FC2 | 1.032 |
| Q | 0.299 |
| K | 0.299 |
| V | 0.299 |
| O | 0.296 |
| Convolutions | 0.117 |
| GELU | 0.076 |
| Add | 0.074 |
| Normalisation | 0.066 |

Q/K/V/O total1.193s; attention and both FFN projections remain larger costs. Individually fenced host submission/wait timing changes synchronisation, so these values identify remaining targets rather than establish matched workflow throughput. No new Q/K/V/O kernel was implemented in this window.

## Verification and isolation

`make model-layout-check host-build host-vet host-test docs-check`, whole-tree race tests, independent marked ARM64/RISC-V builds, latest affected race/vet/tests and documentation checks pass. Existing30 shaders validate/regenerate with normalised identity. No shaders changed, and this does not grant new numerical/feature admission. No independent acoustic/review approval is claimed.

The Intel runner retained CPU4/8GiB/no-swap, heap target4GiB, physical render node, no network/read-only root, Qwen idle and host available memory≥6GiB guards. Qwen LAN/Gemma/services/defaults/resources were untouched. Native invocations retained≤120s test deadlines. Outer240s tool timeouts did not imply completion: guards continued until actual container exit0/noOOM, and containers were removed before reuse or release.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-combined-exact-20261001/) contains raw baseline/combined/expanded JSON, settings, profile samples/totals, actual container states, full build/race/cross logs, shader gate, comparison script, runner and provenance. The combined change only selects existing qualified kernels explicitly.

## Remaining objective

The native Go original-speed goal remains unmet. Continue measured structural work, especially attention/FFN and Q/K/V/O dataflow, without broadening failed activation quantisation or sacrificing expanded word-timing parity. Independent original acoustic/timing quality, matched Vulkan+flash+VAD workflow latency, long-form/resume and fault coverage remain open. This checkpoint is not a deployment or default selection.
