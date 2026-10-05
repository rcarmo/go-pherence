## Pocket TTS pruning metadata reservation

`PruneTape` now counts its admitted dependency closure and reserves exact result tensor, command and output capacities once. Repeated result-slice growth was the largest allocation site in the complete tiny private LSD row's host profile. Closure, stable remapping, command order, arithmetic, independent metadata ownership and read-only constant borrowing are unchanged.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference is unchanged at eight paths and 469 insertions. NVIDIA/Vulkan were disabled. No native GPU, released-model/audio, recovery or publication run occurred.

## Admission and ownership

The full source tape passes validation before roots or dependency closure are considered. Malformed dead commands, uninitialised tensors and over-budget sources still fail. Requested roots must be declared outputs. The reverse traversal marks the same live inputs and retained commands; its command count and the live-tensor count only size the destination metadata.

Result tensor order, command order and duplicate-root order remain stable. Dead IDs map to -1. Shape lists and remapped operand lists still receive independent owned copies; constant values remain borrowed read-only. Constant-only graphs retain nil commands and execute no dispatch. Both source and result validation remain, with unchanged byte and 4096-tensor/command ceilings.

## Alternating allocation samples

Preserved before/after test binaries ran alternately, five samples per fixture with 100 iterations each. Settings: `GOMAXPROCS=2`, offline Go, `-p=1 -parallel=1`, NVIDIA/Vulkan disabled. Warm-up and fixture construction occur outside the timed regions. Raw logs and binaries are retained in the evidence directory; binaries are excluded from the download bundle.

### Isolated pruning

The deterministic vector chain has one independent dead branch. Pruning performs metadata construction and admission without a driver or numerical execution.

| Retained commands | Before B/op | After B/op | Before allocations | After allocations |
| --- | ---: | ---: | ---: | ---: |
| 1 | 528 | 336 | 10 | 8 |
| 128 | 53,048 | 25,352 | 283 | 269 |
| 1,024 | 401,224 | 205,976 | 2,081 | 2,061 |

At 1,024 commands, allocation falls **48.7%**. Median times were 566.3 -> 547.7 ns, 24.92 -> 23.93 us, and 115.36 -> 103.30 us respectively. Samples are noisy; this record establishes allocation reduction and does not establish stable latency or native speed.

### Complete tiny mock LSD row

| Path | Before median B/op | After median B/op | Before median allocations | After median allocations |
| --- | ---: | ---: | ---: | ---: |
| Public | 2,313,203 | 1,934,151 | 8,912 | 8,864 |
| Private | 2,479,500 | 2,100,422 | 8,978 | 8,931 |

Private complete-row allocation falls **15.3%**, about 379 KB/op. Allocation counts vary by one between samples; about 48 allocations disappear. Public medians were 1.250 -> 1.162 ms, with ranges 1.218--1.341 / 1.151--1.279 ms. Private medians were 1.340 -> 1.286 ms, ranges 1.305--1.368 / 1.234--1.473 ms. Overlapping ranges limit latency conclusions. The mock row includes construction, admission, pruning, interpretation, downloads and accumulation, not native device execution.

### Attribution and retention

Separate 20-row `-memprofilerate=1` runs measure attribution without treating instrumented timing as performance evidence. `PruneTape` flat allocation falls **13,248.06 -> 5,105.75 KB**, about 61.5%. Cumulative allocation falls 13,403.44 -> 5,261.12 KB. Initial flow graph command/tensor growth is now a larger cost.

Post-GC total in-use profiles measured 48.45 / 106.98 KB. The after sample includes a 64 KB `backends/simd/runtime.init.func1` allocation absent from the before profile, while regex scratch also differs. Neither profile reports retained `PruneTape` metadata. These setup/profile differences prevent a total retained-memory improvement claim; process peak RSS and released-size/native allocator peaks are unmeasured. No cache or longer-lived destination storage was added.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-prunemetadata-20261002`.
Gate script: `/workspace/tmp/pocket-prunemetadata-gate.sh`.
Final window: **2026-10-02 08:33:41--08:35:00 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 220 top-level + 300 subtests | 0 |
| Selected tests, race detector | 22 top-level + 30 subtests | 0 |
| Backend and affected CPU/mock regressions | 618 top-level + 1,178 subtests | 1 |

There are zero failures. The broad regression selector includes existing `TestTrainingRandomMPFRComparisonDiagnostic`, which skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. That diagnostic did not run and is not a passed gate. Its execution restrictions are unchanged.

New tests cover 1/128/1,024-command retained closures, exact capacities, stable ordering/duplicate roots, owned metadata mutation, deliberately borrowed values, dead branches and nil constant-only commands. Existing tests cover full-source malformed/budget/order admission, all parameter roots including gradients independent of dead parameters, and failed bound/unbound construction/run/download/close/accumulation retention.

Independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step regressions pass unchanged tolerances. Objective endpoints, padding, all gradients/P/M/V/EMA, clipping, checkpoint/resume and device tape/download accounting pass. Vet includes `backends/training`, Pocket TTS and pockettrain; host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass.

`PruneTape` has **97.9% scoped statement coverage**. Its final-validation error return is unexercised: a valid admitted closure preserves the source invariants. Backend package coverage is 86.8%, Pocket TTS 68.4%, pockettrain 14.4% under this selector; these do not establish native coverage.

### Review and limits

A mathematical judge found no closure, alias or budget issue under the stated counting and copying invariants, with constant values still immutable. It did not inspect source or execute native code. Independent source approval, native resources/performance and voice/listening acceptance are open. Holds and numerical thresholds are unchanged.

## Next work

Initial graph command/tensor metadata growth still dominates the host profile. Audio-hidden transformer/flow handoffs, prediction admission, dense embedding VJP, EOS scalar/diagnostic synchronisation and full snapshots/live host copies still require implementation work. Continue bounded offline changes before authorised native qualification.
