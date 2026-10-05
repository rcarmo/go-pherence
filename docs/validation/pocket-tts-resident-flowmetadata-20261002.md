## Pocket TTS flow command metadata reservation

Flow training builders now reserve command storage at forward entry and before reverse emission. IDs, tensor construction, numerical operations, append order, public ownership and replay scopes are unchanged. This removes repeated command-slice growth identified by the previous host allocation profile.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. NVIDIA/Vulkan were disabled. No native GPU, released-model/audio, recovery or publication run occurred.

## Bounded reservation

Forward capacity follows the fixed topology: 35 base commands, 23 per time module, 39 per residual block and two device-time commands per module. Sixteen commands of bounded caller composition headroom are added. Arithmetic saturates incrementally at the existing 4096-command ceiling.

Before reverse emission, `reverseCommandCount` simulates only adjoint presence in a per-call boolean array. It starts from the actual seeded adjoints, visits AD nodes in the same reverse order and counts emitted commands plus joins for already-present gradients. It does not execute math, add tensors, mutate adjoints or change any IDs. The existing reverse emitter remains responsible for all operations and numerical seeds.

The reserved command capacity is the existing forward length plus counted reverse commands and 64 trailing slots, capped at 4096. Those slots avoid immediately regrowing for time roots and caller finish commands. Tensor metadata is unchanged. No persistent cache or shared mutable scratch is introduced. Every actual command/tensor still passes the original complete-source shape, finite and byte admission; capacity is not a numerical budget or a substitute for validation. If estimates cease to match a future topology, append growth remains safe and the existing graph ceiling still applies.

## Measured attempts

Two rejected strategies are retained in the evidence directory. Conservative reverse tensor/command bounds increased private-row allocation by about 33 KB. Exact reverse reservation with too little trailing capacity increased allocation by about 176 KB as subsequent appends grew the slice again. Neither was accepted. Removing tensor reservation and reserving forward commands at builder entry produced the retained result.

The count test initially requested one block from a two-block fixture without trimming it, and a saturation test initially had too few synthetic nodes to reach 4096. Both assertions were corrected before final gates. Numerical tolerances did not change.

## Alternating complete-row samples

Before/after binaries ran alternately for five samples per path, 100 warmed mock rows per sample, with preparation outside measurement. Offline Go, `GOMAXPROCS=2`, `-p=1 -parallel=1` and both GPU disable flags were fixed.

| Path | Before median B/op | After median B/op | Before median allocations | After median allocations |
| --- | ---: | ---: | ---: | ---: |
| Public tiny row | 1,934,145 | 1,719,219 | 8,864 | 8,836 |
| Private tiny row | 2,100,446 | 1,885,464 | 8,930 | 8,902 |

Private complete-row allocation falls **10.2%**, about 215 KB and 28 allocations per call. Public allocation falls 11.1%. Public timing medians were 1.151 / 1.150 ms, ranges 1.130--1.202 / 1.091--1.247 ms. Private medians were 1.265 / 1.223 ms, ranges 1.215--1.333 / 1.222--1.342 ms. Overlapping ranges limit latency conclusions; no reliable native speedup is established.

Separate bounded 20-row `-memprofilerate=1` attribution runs allocated 46,056.66 -> 41,333.93 KB. Flat allocation in `flowTapeGraph.command` falls from 11,460.94 KB to zero; the destination storage is now charged to forward construction and reverse reservation. Flow-builder cumulative allocation falls **21,238.31 -> 16,515.27 KB**, about 22.2%. Tensor allocation and pruning remain large callers.

Post-GC in-use totals were 42.91 / 42.87 KB, essentially unchanged. No retained metadata cache exists. These bounded profiles do not establish process peak RSS, released-size/native allocator memory or sustained latency.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-flowmetadata-20261002`.
Gate script: `/workspace/tmp/pocket-flowmetadata-gate.sh`.
Final window: **2026-10-02 08:50:20--08:51:27 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 100 top-level | 0 |
| Selected tests, race detector | 10 top-level | 0 |
| Backend and affected CPU/mock regressions | 621 top-level + 1,178 subtests | 1 |

There are zero final failures. The broad regression selector includes the existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic`, skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. That diagnostic is unrun, not passed, and its restrictions are unchanged.

Tests compare counted reverse commands with actual emitted commands for host/device time and 1/2/4 residual blocks. They exercise all AD node kinds, repeated adjoint joins, sufficient-capacity reuse, ceiling saturation and independent command-header ownership. Tensor/output headers and adjoint maps remain unchanged by counting/reservation. Existing independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step references pass unchanged tolerances, including objective endpoints, padding, all gradients/P/M/V/EMA, clipping, checkpoint/resume and terminal retention. Tiny tape/download accounting passes unchanged.

All three new functions in `training_flow_metadata.go` have **100% scoped statement coverage**. Backend package coverage is 86.8%, Pocket TTS 68.5%, pockettrain 14.4% under this selector; those figures are separate from native execution. Vet includes all three affected packages; host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. No kernel or scope changed.

### Review and limits

A mathematical judge found no arithmetic/alias/budget concern under the stated capacity-only invariants, highlighting seed simulation and capacity arithmetic for scrutiny. Tests compare counts against the actual seeded emitter and forward arithmetic saturates at the graph ceiling. The judge did not inspect source or execute native code. Independent source approval, native resources/performance and voice/listening acceptance are open. Holds are unchanged.

## Next work

Flow tensor/operand metadata, pruning and selected-result augmentation remain host allocation costs. Audio-hidden transformer/flow handoffs, prediction admission, dense embedding VJP, EOS scalar/diagnostic synchronisation and full snapshots/live host copies still require implementation work. Continue bounded offline changes before authorised native qualification.
