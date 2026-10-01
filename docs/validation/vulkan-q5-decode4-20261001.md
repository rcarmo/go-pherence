# Ordered-F32 Q5 decode4 and projection trials — 1 October 2026

Four-lane Q5 block decoding improves pinned FC1/FC2 kernel medians by6.7–6.8% and fresh trained request medians by1.3–2.2%. Stored values, activations and reduction order are unchanged; all expanded outputs and complete hidden values remain bit-exact. The mode is explicit, with no default or service changes. Original-engine speed/quality acceptance remains unmet.

## Retained candidate

The existing grouped Q5 shader assigns8 lanes per32-value block; each lane reads the scale/high-bit word and decodes2 packed bytes into4 F32 shared values. The candidate assigns4 lanes: each reads the same scale/high-bit word and one packed32-bit low-nibble word, decoding4 bytes into8 shared values. Both write the same32 dequantised F32 values into the same shared locations.

The64×64 output tile, K32 iteration,16 accumulators/lane, ordered multiply/add expression, bias, shared memory and barriers are unchanged. No activation quantisation, scale rounding, arithmetic rewrite or F16 feature is introduced. Original22-byte Q5 blocks still repack losslessly into24-byte aligned blocks.

`NewVkLinearQ5Decode4SetStream`, `NewVulkanEncoderOriginalQ5Decode4` and benchmark arm `vulkan-original-q5-decode4` are separate explicit selections. The ordinary streamed set and eight-lane shader stay unchanged. The streaming helper now accepts a private shader choice; integer-dot still uses its existing separately enabled admission. Metadata, original-value checks, callback ownership and rollback contracts are shared without widening their gates.

Device/shader contract stays16×16 local size,16KiB shared memory,4 buffers,12-byte push constants. Native model weights/scratch/plan/stage stats match baseline. The closed shader inventory now has30 entries; no opcode/feature admission changes.

## Native operator evidence

Synthetic trials use deterministic F32 input/bias and original-format Q5 blocks covering positive/negative/subnormal scales. A33×96×63 tail case and full1500×1280×1280 /1500×5120×1280 shapes compare every output bit with the retained grouped kernel. Five alternating-order executions per arm and guard checks pass.

| Q5 full shape | Eight-lane median (ms) | Four-lane median (ms) | Change |
|---|---:|---:|---:|
| 1500×1280×1280 | 10.575 | 9.853 | −6.82% |
| 1500×5120×1280 | 33.844 | 31.472 | −7.01% |

Pinned original turbo Q5 file SHA256 `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`, with actual layer0 activation files captured in [integer-dot qualification](whisper-integer-dot-ffn-20261001.md), provides a second gate. Five alternating-order samples include every full output comparison, not selected entries:

| Pinned layer0 tensor | Baseline median (ms) | Decode4 median (ms) | Change | Output values checked |
|---|---:|---:|---:|---:|
| FC1 | 32.856 | 30.656 | −6.70% | 7,680,000 |
| FC2 | 33.894 | 31.591 | −6.80% | 1,920,000 |

Latest embedded-operator rerun passes the same pins/output/guard/cleanup checks. These operator timings exclude loading/construction; whole requests are measured separately.

## Rejected structural variants

Before changing the decoder-lane assignment, corrected rectangular tiles, explicit accumulator indexing/unrolls, transposed/padded shared weights, K64 tiles and compiler unroll hints were tested. All valid final shapes pass exact output checks, but they are slower. Five alternating-order samples per arm are retained; the following range spans tested square-F32 and square/FC2-Q5 shapes:

| Variant family | Median change | Decision |
|---|---:|---|
| Explicit four/eight-channel accumulator expressions | +28–45% | Reject |
| Explicit128×32 /32×128 output tiles | +46–89% | Reject |
| Explicit transposed weight reads | +36–49% | Reject |
| Baseline-loop32×64 /64×32 tiles | +16–26% | Reject |
| Baseline-loop32×32 tile | +52–59% | Reject |
|64×64 tile with K64 | +61–72% | Reject |
| Baseline-loop shared-weight transpose | +0.64% F32; +7–8% Q5 | Reject |
| Shared weight stride33 | +37–54% | Reject |
| K/all-loop compiler unroll hints | +28–36% | Reject |
|16/32 lanes per Q5 block | +7–8% /+36–37% | Reject |

The first two rectangular F32 generator versions had mismatched shared loader extents. The32×128 case drifted and the128×32 loader wrote beyond its smaller weight tile; these versions are invalid and excluded from accepted timings. They are retained in separately labelled invalid directories. After splitting X/W loaders with correct extents, all candidate outputs pass. No invalid run is presented as arithmetic parity or speed evidence.

The initial explicit-FMA trial was also slower; final valid structural runs use the original multiply/add expression. Corrected versions are separately archived. No tolerance widening or shader-admission expansion occurred. Rejected temporary harnesses/shaders are not in runtime packages; only decode4 is retained.

## Five-repeat request comparison

Physical Intel Iris Xe, four CPU threads, original packed Q5 FFN with F32 activations, other original stored values widened to F32, retained F32 key32 score-ILP flash attention and host decoder/alignment. Decode4 alone is compared with `vulkan-original-q5-source`; it does not silently combine the previous attention-unroll mode. Each fresh baseline process precedes the candidate process, with five requests including the first. No timed sample is discarded; order is not randomised.

| Fixture | Fresh baseline median (s) | Decode4 median (s) | Change | Every returned output |
|---|---:|---:|---:|---|
| JFK English | 6.919 | 6.767 | −2.20% | Exact |
| MINDS Portuguese row0 | 6.852 | 6.720 | −1.93% | Exact |
| MINDS French row0 | 6.343 | 6.264 | −1.25% | Exact |
| JFK native Silero VAD + words | 8.094 | 7.938 | −1.92% | Exact windows/segments/word times |
| Two speech groups, VAD + words | 16.282 | 16.000 | −1.73% | Exact windows/segments/word times |

Model loading, native preparation and five-request full-arm totals are separate in raw JSON. A phase/operator win is not called original-engine workflow acceptance. No original engine run was collected in this window; earlier original timing/quality gaps remain.

Four additional recordings ×five candidate requests pass exact baseline-output gates: MINDS PT rows1/2 (including43.76s), podcast20s without VAD, and podcast20s with native Silero VAD/word alignment. Their baselines come from earlier expanded-quality evidence, so these are output/timing gates, not fresh speed pairs. The candidate does not reproduce the FC1 quantisation word change or20ms alignment shift. Five silence/VAD/word requests return no windows.

## Hidden values, lifetime and verification

Pinned complete encoder synthetic mel yields1,920,000 bit-identical hidden values. Memory/stage/plan stats remain unchanged. Three cancellation points, drain if needed, bitwise reuse and complete native cleanup pass; latest run records1276 checkpoints. Model-free decode4 constructor/close tests validate shader ABI, copied payload, cancellation/nil-reader, device shared-memory bounds and unchanged default selection. Existing reader-error/extent/nonfinite-scale/panic rollback tests still pass through the shared stream helper.

`make model-layout-check host-build host-vet host-test docs-check`, whole-tree race, independent marked ARM64/RISC-V builds and latest affected race/vet/tests pass. All30 stored/embedded shaders validate and regenerate with normalised identity; checker tests pass. No separate numerical approximation budget was introduced: candidate admission is exact stored-value/native-baseline parity.

The isolated runner retained CPU4/8GiB/no-swap, heap target4GiB, physical Intel render node, no network/read-only root, Qwen idle and host available memory≥6GiB guards. Qwen LAN/Gemma/services/defaults/resources were untouched. Native test deadlines were≤120s. An outer timeout left the groups/gate container running; guards were continued until actual exit0/noOOM, then containers removed before reuse/release. No independent review approval is claimed.

[Hashed evidence](../../benchmarks/speech-foundations/vulkan-q5-decode4-20261001/) retains corrected and invalid prototypes, exact diagnostic harnesses, pins, full native/request JSON/logs/states, expanded gates, fresh pairs, comparison script, runner, shader gate and build/race/cross evidence. The change is explicit and limited; no rejected tiling prototype remains in runtime.

## Next objective

Measure an explicitly combined exact decode4+attention-unroll mode rather than add their percentages. Continue measured F32 Q/K/V/O/attention work and qualify every new candidate on the expanded set. Independent original word timing/acoustic accuracy, long-form/resume and fault coverage remain open. Matched native Go Vulkan+flash+VAD speed against original is still not verified; the overall goal remains unmet.
