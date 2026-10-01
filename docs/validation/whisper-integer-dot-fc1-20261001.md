# Portuguese timestamp attribution and FC1-only integer-dot — 1 October 2026

The separately named FC1-only integer-dot candidate preserves all retained English/Portuguese/French and native-VAD/word outputs across five repeats, while reducing request medians by3.8–5.3%. FC2 remains ordered packed-Q5/F32. No decoder clamp, fallback, special timestamp rule or wider tolerance is used. Defaults/services stay unchanged. This is limited-fixture qualification, not independent acoustic accuracy or original-speed acceptance.

## Closing timestamp attribution

A test-only recorder copies raw logits after the actual `ForwardToken` call, then replays the unchanged suppression and timestamp mask on those copies. The verified recorder checks replay selections against tokens actually consumed by the decoder. It never changes the live logits/state, generation rules or returned segments. Timed inference does not use the recorder.

With the original pinned Q5 values and Portuguese MINDS fixture, both modes generate the same initial timestamp and25 content tokens. Only the closing timestamp at generated step26 differs. Both candidate endpoints survive the same mask; their raw and masked scores match:

| Arithmetic | 7.36s score | 9.04s score | Selected endpoint |
|---|---:|---:|---:|
| Baseline packed-Q5/F32 | 8.520127 | 8.196755 | 7.36s |
| Full FFN integer-dot | 8.386263 | 8.707090 | 9.04s |
| Full FFN, separate block multiply/add | 8.387697 | 8.653187 | 9.04s |
| FC1-only integer-dot | 8.466670 | 8.228115 | 7.36s |
| FC2-only integer-dot | 8.386379 | 8.545753 | 9.04s |
| First16 layers, both FFN projections quantised | 8.538029 | 8.606315 | 9.04s |
| Last16 layers, both FFN projections quantised | 8.619143 | 8.207121 | 7.36s |

The separate multiply/add trial mirrors the compiled original MMQ block accumulation but does not resolve the endpoint regression. Original source at `c44b60b8053bbf2a5c1e014f11323fb3f2485177` returns `dW*(integerSum*dQ-16*sQ)` and adds it into F32 accumulators. Its existing `matmul_q5_0_q8_1_fp32` binary disassembly shows separate `OpFMul`/`OpFAdd`, rather than the explicit block FMA in the Go candidate. The trial was reverted; retained embedded shader bytes remain identical to the published full-FFN candidate.

These ablations implicate FC2 activation quantisation in this fixture, especially earlier encoder layers; they do not prove a universal single-layer cause. Changing FMA alone is insufficient. Original Q/K/V/O and attention also have different precision boundaries, so this attribution does not establish whole-original arithmetic equivalence.

## Explicit FC1-only candidate

`NewVulkanEncoderOriginalQ5IntegerDotFC1` and benchmark arm `vulkan-original-q5-integer-dot-fc1` are explicit experimental selections. They require `VulkanInitIntegerDot` and the pinned original Q5 checkpoint. FC1 uses Q8_1 activation quantisation and the existing integer-dot projection; FC2 explicitly chooses the retained ordered-F32 activation kernel. Both access the same immutable original packed weight arena through separate typed owner kernels.

The hybrid owner is not a device-unsupported fallback. Callers choose both kernels during construction; the unsupported integer-dot device is still refused. The streamed weight reader/metadata/value gates are unchanged. Packed-only benchmark loading omits CPU FFN widening and clears the CPU encoder after successful native construction. Other weights, attention, decoder and word alignment remain the existing F32/CPU paths.

Native weight bytes stay1,184,890,880. Scratch is95,856,000 bytes, +2.16MB over baseline (only FC1 Q8 scratch);34 plans and422 stages, +32 stages over baseline. The full-FFN candidate remains separately available for diagnostics and retains its recorded timing failure; defaults are unchanged.

The private projection-selection seam exists only for numerical attribution and refuses packed-only loading or non-integer-dot modes. It cannot select arbitrary defaults. Model-free tests cover hybrid F32 stage/offset/shape, alias and index refusal, disabled/closed owners, both kernels' in-flight retention, cancellation during construction, reader/extent failure and quantiser/hybrid-kernel rollback. Pinned native tests compare hybrid F32 outputs against the existing grouped-Q5 operator bitwise for all layer0 FC1/FC2 output values.

## Five-repeat whole-request diagnostics

Physical Intel Iris Xe, CPU4, pinned original-Q5 stored values, language-specific greedy decoding and F32 key32 score-ILP flash attention. Candidate and fresh baseline are separate guarded processes from the same window; process order is candidate then fresh baseline, not randomised. Five samples include the initial request; none are discarded. Returned windows/segments/tokens/words are compared structurally across every repeat.

| Fixture | Fresh baseline median (s) | FC1-only median (s) | Change | All returned outputs |
|---|---:|---:|---:|---|
| JFK English | 6.916 | 6.563 | −5.10% | Exact |
| MINDS Portuguese | 6.892 | 6.553 | −4.91% | Exact, endpoint7.36s restored |
| MINDS French | 6.360 | 6.023 | −5.30% | Exact |
| JFK native Silero VAD + words | 8.097 | 7.793 | −3.76% | Exact speech/audio windows, segments and word times |
| Two speech groups, VAD + words | 16.338 | 15.670 | −4.09% | Exact speech/audio windows, segments and word times |

Five silence/VAD/word requests return no windows. Preparation/loading/full-arm time is separately retained. JFK five-request full-arm38.250→36.971s; Portuguese38.167→36.750; French35.540→33.958; JFK VADwords44.378→42.690; groups85.552→82.205. These are not matched original-engine cold workflow timings; no original-speed win is claimed.

Compared with the full-FFN candidate, this chooses less activation quantisation, recovers Portuguese timing and retains about half of the stage/request gain. The trade-off is explicit across all32 FC1 layers, not a fixture/language special case.

## Hidden drift, cancellation and gates

On the retained synthetic mel encoder diagnostic,1,919,996 of1,920,000 hidden values differ from the F32 baseline; maximum absolute drift2.38079, mean0.008828. This is smaller than full-FFN drift but is still not bitwise-F32 equivalence or an independent accuracy threshold. Three cancellation checkpoints, native drain if needed, candidate-bitwise reuse and complete allocation cleanup pass. Latest hybrid run records1304 checkpoints and2.16MB extra scratch.

Isolation retained CPU4/8GiB/no-swap, Go heap target4GiB, no network/read-only root and physical Intel render node. Qwen idle and host available memory≥6GiB guards stayed active. Qwen LAN/Gemma/services/resources remained unchanged. Every native invocation had a deadline≤120s. Outer tool timeout did not imply drain: the gate container was guarded until actual exit0/noOOM and then removed before later work.

Full build/vet/tests, layout/docs and whole-tree race gates pass. ARM64/RISC-V builds and latest affected race tests additionally have independent visible pass markers. All28 retained shaders plus the two optional integer-dot regeneration/validator gates pass; the unfused temporary shader is not retained in the backend. The final recorder verifies replay against actual decoder consumed tokens. No independent acoustic/review approval is claimed.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-integer-dot-fc1-20261001/) retains raw timestamp traces, ablation shader/source and original MMQ binary/disassembly, fresh baseline/candidate JSON, construction/native tests, phase logs, runtime states, runner, pins and comparison totals. Earlier [full-FFN qualification](whisper-integer-dot-ffn-20261001.md) retains independent original Q8 block-byte parity and the failed endpoint gate.

## Remaining objective

Matched original Vulkan+flash+VAD workflow speed, independent acoustic/word-timing accuracy, long-form/resume and fault coverage remain open. Before broader quantised projection work, profile the FC1-only candidate and independently qualify its accuracy; Q/K/V/O and attention remain substantial F32 costs. Do not infer quality from five short fixtures or treat their exact outputs as a general numerical promise.
