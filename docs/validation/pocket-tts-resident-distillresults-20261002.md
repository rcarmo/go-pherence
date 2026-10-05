## Pocket TTS private distillation results

`ptx-resident-distillresults-hybrid-v1` and `vulkan-resident-distillresults-hybrid-v1` reduce the training step's distillation downloads to raw condition/target gradients, loss statistics and a scalar finite-check witness. Public `EvaluateDistill` and `LSDDistillTrainingBackend` still return noise and time derivatives. Earlier scopes keep their download policy.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index still contains eight paths and 469 insertions. NVIDIA/Vulkan were disabled throughout these gates; no model/audio, native GPU, recovery or publication run occurred.

## Contracts and finite witness

The private call is selected only from the training step and only in the new scopes. It returns no fake noise/time arrays. Raw condition/target/logvar derivatives and statistics pass into the existing step reduction, which applies objective and valid-row scaling once. Primary parameter roots accumulate exactly once at the unchanged parameter scale, including zero scales. Endpoint direct parameter, condition and time gradients never accumulate.

The original downloads checked noise/time derivatives for finiteness. Removing those checks would admit an invalid computation. Two runtime zero-weight linear reductions retain those derivative roots and return a summed scalar witness: finite F32 values produce zero, NaN/Inf produce NaN. The zero weights are tensor inputs; the existing strict-FP kernels load and multiply them without constant folding. No square sum can overflow on finite maximum derivatives. The host rejects nonfinite or nonzero witnesses. This exact-zero check enforces a diagnostic contract, separate from numerical parity or signal acceptance.

The witness uses existing linear/vector/reshape commands; no kernel changed. Native execution has not verified zero-product NaN propagation on the held hardware. CUDA source uses strict arithmetic compilation and Vulkan source uses `precise` products/accumulators; qualification under the pinned runtime is still required.

Complete source/binding/byte admission precedes result selection and is repeated after adding the witness. The graph ceilings and gradient inventory are unchanged. Selected outputs must have the correct extent, be finite, disjoint from each other and independent of every caller input. Square statistics must be nonnegative. Failed source/witness/result admission makes the backend terminal, blocks retry/checkpoint/ordinary close and keeps uncertain owners reachable.

## Tiny complete-step accounting

| Measure | Stepresults | Distillresults |
| --- | ---: | ---: |
| Tapes | 12 | 12 |
| Boundary tensor downloads | 44 | 42 |
| Downloaded result/gradient elements | 174 | 168 |
| Commands | 3,683 | 3,697 |
| Conservative logical tape bytes | 132,756 | 132,908 |
| Summed planned owned tape bytes | 27,960 | 27,960 |

Two active distillation rows each replace two arrays with one scalar witness. Accumulation counts are unchanged. The witness adds fourteen commands and 152 logical bytes across this fixture. Planned owned storage is unchanged after zero pooling/lifetime planning. These counters cover tape downloads; diagnostic/full-snapshot transfers are unchanged. Native latency, allocator peak and transfer-inclusive throughput are unmeasured, so this scope is opt-in.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-distillresults-20261002`.
Gate script: `/workspace/tmp/pocket-distillresults-gate.sh`.
Final window: **2026-10-02 07:22:59--07:23:52 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 190 top-level + 310 subtests |
| New tests, race detector | 19 top-level + 31 subtests |
| Affected CPU/mock regression | 409 top-level + 674 subtests |

Final logs contain zero failures/skips. Tape tests compare all selected derivatives/statistics with the full distill graph at boundary/interior times, check unchanged bindings and exact logical ceilings, and reject invalid dead sources before pruning. Separate finite-witness tests induce positive/negative Inf and NaN from admitted finite graph sources and accept positive/negative maximum finite derivatives without overflow.

PTX/Vulkan host mocks cover public/private parity, raw input derivatives at parameter scales 0/.25/1, primary-only once-per-root accumulation, public LSD noise/time outputs, private step routing, objective endpoints, empty/nonempty voice, padding, clipped multistep parameter/M/V/EMA updates and checkpoint/resume/cross-scope rejection. Malformed shapes/statistics, output/input aliases, invalid witness values, inactive/changed owners/scopes and complete-source-versus-witness byte ceilings fail terminally. Every graph construction failure blocks retry and checkpoint.

Existing independent CPU/F64 and pinned PyTorch mixed-AD/minimal-LSD/full-step regressions pass with unchanged tolerances. All five functions in `training_distill_results.go` have **100% scoped statement coverage**. Whole-package/native coverage is not established. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks passed.

### Development corrections and review

Retained development logs record a finite-witness test that initially placed NaN in an input (correctly rejected by admission), mismatched fixture conditioning width, an incorrect element-count expectation, and update assertions before non-flow roots were marked ready. Those tests now use finite sources to induce invalid results, model-sized conditioning, measured counts and explicit zero non-flow branches. A follow-up hook signature compile error was corrected. No numerical tolerance changed.

A read-only source judge timed out after 60 seconds and supplied no review. Independent source approval and native/resource/voice acceptance are open. No hold was released.

## Next boundaries

Forward prediction/admission, transformer-to-flow handoffs, forward EOS scalar, host graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation and full snapshots/live host copies still require work. Older builder/evaluator coverage gaps stay tracked separately. Continue bounded offline implementation until Rui and `@default` explicitly coordinate native execution and publication.
