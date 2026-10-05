# Compact projection scheduling trials — 1 October 2026

All 17 variants are rejected. Shared-memory rotations, workgroup aspect ratios, narrower Q5 decoder assignments, double-buffered Q5 tiles and smaller F32 K tiles preserve tested output bits but increase kernel time. Swapping the Q5 output-loop nesting is almost tied synthetically and 0.6–1.3% slower on pinned trained inputs. No model mode, request gain or default change is introduced.

## Scope

The baseline is the accepted four-lane original-Q5 decoder and the ordinary F32 register-tile64 projection. Both use a 64×64 output tile, K32 steps, 256 lanes and 16 ordered F32 accumulators per lane. This window does not repeat the previously rejected rectangular output tiles, shared stride33, transpose layouts, K64 tiles or compiler unroll hints.

Original Q5 weights remain losslessly packed into 24-byte blocks, with the source F16 scale, high-bit word and four nibble words unchanged. No prepared-byte expansion, activation quantisation, optional device feature, arithmetic tolerance or shader admission change occurs. Every output still consumes K in increasing order with the original multiply/add expression and bias addition.

## Q5 variants

Thirteen variants compare against the retained decode4 operator:

- `wrotate1` and `wrotate2` rotate each shared weight row's K index by the row's lane-column index, or twice that index, modulo32. `bothrotate` also rotates shared activation rows. Allocation size and weight values are unchanged; matching reads undo the indexing transformation.
- `colmajor` visits four columns before four rows inside each K step, instead of rows before columns. Each output's accumulation order is unchanged. `colmajorrotate` combines the loop swap with weight rotation.
- `local32x8` and `local8x32` keep 256 lanes and the 64×64 output tile, but distribute each lane's 16 outputs as 8×2 or 2×8. The `col` forms swap the nested output loops. Load/decode bounds and output mapping use the adjusted local geometry.
- `decode1` and `decode2` assign one or two lanes per compact Q5 block, loading four or two packed nibble words per participating lane. Scale/high bits are reused across more reconstructed values, but fewer lanes perform decoding.
- `pingpong` and `pingpongcol` alternate between two shared X/W tiles. Shared storage increases from16 to32KiB, removing the trailing compute barrier. The next load uses the other tile; its pre-compute barrier completes prior reads before that prior tile can be reused two steps later. K order and arithmetic are unchanged. This is not a GPU/host overlap or asynchronous-copy claim.

Each variant runs a 33-row/K96/N63 tail and full1500-row FC1/FC2 shapes K1280/N5120 and K5120/N1280. Every output bit and both guard regions match baseline. Five alternating-order timing samples per arm include the first. Final native rerun covers all13×3 shape cases.

| Final Q5 variant | FC1 change | FC2 change | Decision |
|---|---:|---:|---|
| Weight row rotate1 | +19.28% | +18.37% | Reject |
| Weight row rotate2 | +28.84% | +27.87% | Reject |
| X/W row rotate1 | +21.83% | +20.64% | Reject |
| Column-first output loop | +0.85% | +0.86% | Trained check, then reject |
| Column-first + rotation | +33.86% | +25.78% | Reject |
| Local32×8 | +10.54% | +10.57% | Reject |
| Local8×32 | +4.28% | +5.75% | Reject |
| Local32×8 column-first | +8.37% | +10.07% | Reject |
| Local8×32 column-first | +10.17% | +10.01% | Reject |
| One lane per Q5 block | +41.06% | +39.47% | Reject |
| Two lanes per Q5 block | +32.25% | +32.07% | Reject |
| Two alternating shared tiles | +59.50% | +59.95% | Reject |
| Alternating tiles column-first | +41.26% | +42.38% | Reject |

Separate initial processes reach the same rejection decisions: rotations+19–33%, geometry+5–12%, one/two decode lanes+31–39%, alternating tiles+38–60%, and column-first+0.5–0.7%. Absolute baseline timings rise during the final sequence. GPU clocks/temperature are not controlled or recorded; no cause is assigned. Comparisons alternate locally within each shape, and no samples are discarded or percentages pooled across processes.

## Trained near-tie check

Only `colmajor` warrants trained follow-up; the other synthetic regressions are already substantial. The pinned original turbo Q5 model and retained block0 `ff-norm`/`tmp-ff` inputs match their SHA256 pins. Five alternating samples check all7,680,000 FC1 and1,920,000 FC2 output values on every run.

| Tensor | Decode4 median (ms) | Column-first median (ms) | Change |
|---|---:|---:|---:|
| FC1 | 30.712 | 30.907 | +0.64% |
| FC2 | 31.531 | 31.932 | +1.27% |

Both outputs are bit-identical. There is no useful trained gain and no encoder integration. This diagnostic does not establish an independent scalar projection oracle, acoustic quality, expanded VAD/word outputs or complete hidden-encoder parity. No cancellation/fault injection is added; allocation guards and normal native cleanup pass.

## F32 smaller-K trials

Four F32 variants retain compact source values, the original output tile and K order. K16/K8 steps reduce shared tile allocation to8/4KiB but add barriers/iterations. Their double-buffered forms require16/8KiB and remove the trailing barrier using the same alternating-buffer scheme.

Final native execution covers five shapes per variant: 1×1×1, 17×17×65, 33×97×63, 33×96×63 and1500×1280×1280. Odd-K/output/query tails and complete output bits/guards pass. Five alternating-order timings include the first on the full shape.

| Final F32 variant | K32 baseline median (ms) | Candidate median (ms) | Change |
|---|---:|---:|---:|
| K16 | 14.739 | 21.047 | +42.79% |
| K8 | 12.096 | 17.948 | +48.38% |
| K16 alternating tiles | 12.471 | 16.310 | +30.78% |
| K8 alternating tiles | 12.443 | 17.740 | +42.58% |

Earlier two-shape runs also reject all four. Absolute baseline times differ noticeably between variants/processes; these are isolated operator diagnostics, not end-to-end or original-engine comparisons. No trained F32 projection, model mode or extra accuracy budget is claimed.

## Retained tree and evidence

All17 diagnostic shaders compile and pass offline `spirv-val`; the existing closed Vulkan contract admits their native construction without expansion. The diagnostics execute13×3 Q5 cases,2 trained Q5 tensors, and4×5 F32 cases in final runs. Synthetic shape checks are exact native-baseline comparisons, not independent acoustic/reference acceptance.

The read-only review delegate timed out after60 seconds. No independent review approval is claimed. No observed numerical failure, unsafe loader correction or skipped-native pass occurred in this window. Rejected shader/test sources are removed from runtime and test discovery; [hashed evidence](../../benchmarks/speech-foundations/vulkan-projection-scheduling-20261001/) retains generators, both harnesses, SPIR-V, all raw timings, comparison parser/JSON, pins/settings, runner, container states/exits and retained-tree logs.

The unchanged production-source tree passes `make model-layout-check host-build host-vet host-test docs-check`, affected Vulkan/Whisper race tests and marked ARM64/RISC-V builds. Final documentation checks run after adding this report. No new runtime shader enters the31-shader inventory.

The fresh @llama isolation hold preserves live Qwen LAN and Gemma state. Native execution uses CPU4/8GiB/no-swap, physical Intel GPU, Qwen-idle and host available memory≥6GiB guards; trained and final Q5 runs set heap4GiB. All native deadlines stay≤120s. Container exit0/noOOM records and actual drain—not outer-command completion alone—precede release. Defaults/services/resource allocations remain unchanged.

The original Vulkan+flash+genuine-VAD speed and quality goal remains unmet. The retained fastest exact path is still outputILP attention plus decode4 FFN. The next work needs a freshly matched whole-workflow attribution/acceptance boundary or a materially different qualified kernel approach; none of these scheduling trials narrows the measured original-engine gap.
