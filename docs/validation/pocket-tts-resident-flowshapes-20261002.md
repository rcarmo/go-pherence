## Pocket TTS flow tensor shape arena

Flow graph builders copy tensor shapes into bounded per-graph integer chunks. This removes per-tensor allocations identified in the object profile while preserving mutable, append-isolated shape metadata. Tensor values, parameter borrowing, IDs, command order, arithmetic, public outputs and replay scopes are unchanged.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. NVIDIA/Vulkan were disabled. No native GPU, released-model/audio, recovery or publication run occurred.

## Shape ownership

The previous bounded profile attributed 35,415 objects to `flowTapeGraph.tensor`, mainly one copied shape per tensor. The new `flowShapeArena` belongs to one graph builder. Valid rank-1/2/3 shapes occupy disjoint regions of 256-integer chunks; full-slice expressions set capacity equal to length. Mutating or appending one shape cannot reach another shape or the caller's source. Empty shapes retain nil behaviour, and invalid ranks use the previous independent copy before existing admission rejects them.

Shapes remain independent across graph builds, with no cache or global arena. A shallow copy of an actively appending private graph must start a fresh arena. The existing reverse-count test that copies a graph now resets this field before independently appending. Production composition constructs fresh builders and keeps source metadata read-only.

Retaining one shape can pin at most 256 integers (2 KiB on amd64), a bounded incidental retention tradeoff. Host arena bytes are additional to logical tensor budgets, as existing host metadata was. Complete source validation, byte ceilings and graph ceilings are unchanged.

## Alternating complete-row samples

Preserved before/after binaries ran alternately for five samples per path with 100 warmed mock rows each. Preparation was outside measurement. Offline Go, `GOMAXPROCS=2`, `-p=1 -parallel=1` and both GPU disable flags were fixed.

| Path | Before median B/op | After median B/op | Before median allocations | After median allocations |
| --- | ---: | ---: | ---: | ---: |
| Public tiny row | 1,718,372 | 1,724,655 | 6,595 | 4,462 |
| Private tiny row | 1,884,599 | 1,890,892 | 6,641 | 4,508 |

Private allocation count falls **32.1%**, by 2,133 allocations per complete row. Allocated bytes rise by about **6.3 KB/op (0.33%)** from chunk padding and size classes. Public allocation count falls 32.3% with a similar byte cost. This change targets object churn, not a byte reduction.

Public timing medians were 1.122 / 1.092 ms, ranges 1.102--1.208 / 1.047--1.154 ms. Private medians were 1.222 / 1.162 ms, ranges 1.150--1.277 / 1.116--1.215 ms. Overlapping ranges limit latency conclusions. The row includes graph build, admission, pruning, interpretation, downloads and gradient accumulation; it does not execute a GPU or released model.

Separate bounded 20-row `-memprofilerate=1` attribution runs show flow tensor construction **35,497 -> 1,236 cumulative objects**, about 96.5% less including arena allocations. Flat objects were 35,497 / 900. Post-GC total in-use profiles measured 43.35 / 42.99 KiB, with no observed retained increase in this fixture. These setup-inclusive profiles do not establish peak RSS, released-size residency or native allocator peaks.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-flowshapes-20261002`.
Gate script: `/workspace/tmp/pocket-flowshapes-gate.sh`.
Final window: **2026-10-02 09:08:00--09:09:04 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 120 top-level | 0 |
| Selected tests, race detector | 12 top-level | 0 |
| Backend and affected CPU/mock regressions | 625 top-level + 1,178 subtests | 1 |

There are zero failures. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun and is not a passed gate; its restrictions are unchanged.

Tests cover nil/invalid rank copies, caller mutation, shape writes/appends, chunk boundaries, cross-call shape isolation, retained graph numerical results and concurrent independent builds. Existing reverse-count tests, source ownership/admission, independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step references pass unchanged tolerances. Objective endpoints, padding, all gradients/P/M/V/EMA, clipping, checkpoint/resume, terminal retention and tape/download accounting pass.

The shape-copy helper has **100% scoped statement coverage**. Package coverage is 87.0% for `backends/training`, 68.6% for Pocket TTS and 14.4% for pockettrain under this selector. Vet includes all three affected packages; host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. No kernel or scope changed.

### Review and limits

A delegated invariant judge timed out after 15 seconds and supplied no approval. Independent source review, native accuracy/resources/performance and voice/listening acceptance are open. The byte and bounded-retention tradeoffs are recorded above. Numerical thresholds and holds are unchanged.

## Next work

Flow operand/node metadata and selected-result augmentation still allocate host storage. Audio-hidden transformer/flow handoffs, prediction admission, dense embedding VJP, EOS scalar/diagnostic synchronisation and full snapshots/live host copies still require implementation work. Continue bounded offline changes before authorised native qualification.
