## Pocket TTS flow command operand arena

Flow command construction now copies operand lists into bounded per-graph integer chunks. This removes slice-literal allocations identified by the latest object profile and gives each command its own mutable operand region. IDs, command order/modes/scales, tensor values, gradients, admission and replay scopes are unchanged.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. NVIDIA/Vulkan were disabled. No native GPU, released-model/audio, recovery or publication run occurred.

## Ownership contract

`flowOperandArena` copies each command's inputs into disjoint regions of per-graph 256-integer chunks, with capacity equal to length. Mutating or appending one list cannot reach sibling commands or the source list. Empty lists retain nil behaviour. Lists longer than eight operands use standalone owned copies, also capacity-limited; supported flow commands use bounded arities.

Reverse stages previously shared an input-list backing array within some linear/norm/modulation nodes. Those commands now have independent operand regions. Callers do not rely on mutation through that sharing: the reverse emitter reads the lists, and graph composers already copy operands before remapping. Public tape mutation is therefore isolated more strictly without changing executed values.

The arena has no global cache. Independent and concurrent builds have separate storage. A test that shallow-copies the graph resets both shape and operand arenas before independently appending. Retaining an admitted operand list can pin at most 256 integers (2 KiB on amd64). Host arena storage is additional to logical tensor budgets, as all existing host graph metadata is. Complete source validation and graph ceilings still apply.

The initial invalid-large-list fallback had spare capacity from an append copy and failed the append-isolation test. It now uses a full-slice expression too. No numerical tolerance changed.

## Alternating complete-row samples

Before/after binaries ran alternately for five samples per path with 100 warmed CPU/mock rows each. Preparation was outside measurement. Offline Go, `GOMAXPROCS=2`, `-p=1 -parallel=1` and both GPU disable flags were fixed.

| Path | Before median B/op | After median B/op | Before median allocations | After median allocations |
| --- | ---: | ---: | ---: | ---: |
| Public tiny row | 1,724,641 | 1,742,325 | 4,462 | 3,022 |
| Private tiny row | 1,890,868 | 1,908,565 | 4,508 | 3,068 |

Private allocation count falls **31.9%**, by 1,440 allocations per complete row. Allocated bytes rise by **17.7 KB/op (0.94%)** because operand chunks include padding and reverse stages no longer share list storage. Public allocation count falls 32.3% with a similar byte cost. This change targets object churn and independent ownership, not a byte reduction.

Public timing medians were 1.102 / 1.043 ms, ranges 1.055--1.190 / 1.027--1.134 ms. Private medians were 1.183 / 1.160 ms, ranges 1.107--1.221 / 1.098--1.221 ms. Overlap limits latency conclusions. The complete mock row includes graph construction, admission, pruning, interpretation, downloads and accumulation; it does not execute a GPU or released model.

Separate bounded 20-row `-memprofilerate=1` attribution runs allocated **87,051 -> 63,980 objects**, about 26.5% less including setup. The previously profiled reshape operand-literal allocation disappears, and composition remapping becomes the largest object contributor. Post-GC total in-use profiles measured 42.87 / 42.77 KiB, essentially unchanged. These setup-inclusive totals do not establish peak RSS, released-size residency or native allocator peaks.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-flowoperands-20261002`.
Gate script: `/workspace/tmp/pocket-flowoperands-gate.sh`.
Final window: **2026-10-02 09:14:24--09:15:32 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 160 top-level | 0 |
| Selected tests, race detector | 16 top-level | 0 |
| Backend and affected CPU/mock regressions | 629 top-level + 1,178 subtests | 1 |

There are zero failures. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun and is not a passed gate; its restrictions are unchanged.

Tests cover nil and oversized copies, caller mutation, writes/appends, chunk boundaries, sibling-command independence, cross-call isolation, retained graph numerical results and concurrent builds. Existing source ownership/admission, reverse-count, independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step references pass unchanged tolerances. Objective endpoints, padding, all gradients/P/M/V/EMA, clipping, checkpoint/resume, terminal retention and tape/download accounting pass.

The operand-copy helper has **100% scoped statement coverage**. Package coverage is 87.0% for `backends/training`, 68.6% for Pocket TTS and 14.4% for pockettrain under this selector. Vet includes all three affected packages; host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. No kernel or scope changed.

### Review and limits

A mathematical judge found the copy and capacity-limited ownership invariant sound, with the caveat that intentional input-list aliasing or spare-capacity dependence would be incompatible. Inspected graph emitters/composers read or independently copy operands; no such dependence was found. The judge did not inspect source or execute native code. Independent source review, native accuracy/resources/performance and voice/listening acceptance are open. Byte and retention tradeoffs are recorded above; numerical thresholds and holds are unchanged.

## Next work

Composition remapping and AD node metadata still allocate host storage. Audio-hidden transformer/flow handoffs, prediction admission, dense embedding VJP, EOS scalar/diagnostic synchronisation and full snapshots/live host copies still require implementation work. Continue bounded offline changes before authorised native qualification.
