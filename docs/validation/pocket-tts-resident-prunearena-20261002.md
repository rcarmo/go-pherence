## Pocket TTS pruning shape/operand arena

`PruneTape` now stores copied shapes and remapped operands in per-call integer chunks, removing thousands of small metadata allocations. Every returned slice has disjoint storage and capacity equal to its length; mutation and append cannot reach neighbouring metadata. Full-source admission, dependency closure, stable IDs/order, arithmetic and read-only constant values are unchanged.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. NVIDIA/Vulkan were disabled. No native GPU, released-model/audio, recovery or publication run occurred.

## Ownership and bounded retention

The previous object profile attributed 19,879 allocations to shape copies and 17,395 to operand lists in the bounded private-row run. Result headers were already reserved; the remaining per-list storage was measured separately.

After complete source validation and unchanged closure traversal, pruning counts retained shape/operand integers. Each per-call chunk holds at most 256 integers for admitted metadata, with a smaller final chunk sized from remaining demand. Shape ranks and supported command operand counts are bounded by source validation, so the total count cannot overflow. The arena has no global state or cache.

`take` returns a full-slice expression with capacity limited to length. Shape contents are copied and operands remapped in original order. Each slice is independent for writes and appends, but disjoint slices can share the same chunk allocation. Retaining a single small shape or operand list therefore pins at most 256 integers (2 KiB on amd64), instead of an entire graph-sized arena. This bounded incidental retention is a tradeoff; logical tape byte ceilings account tensor values, not host metadata overhead. Constant values keep their original read-only borrowing contract.

An initial fixed-size arena padded every tiny graph and slightly increased complete-row bytes. It was replaced with demand-aware final chunks before the final gates; rejected sample logs are retained. No numerical threshold changed.

## Alternating allocation samples

Before/after binaries ran alternately, five samples per fixture with 100 iterations each. Preparation and warm-up occur outside measurement. Offline Go, `GOMAXPROCS=2`, `-p=1 -parallel=1` and NVIDIA/Vulkan disable flags were fixed.

### Isolated pruning

| Retained commands | Before B/op | After B/op | Before allocations | After allocations |
| --- | ---: | ---: | ---: | ---: |
| 1 | 336 | 344 | 8 | 5 |
| 128 | 25,352 | 25,464 | 269 | 13 |
| 1,024 | 205,976 | 205,976 | 2,061 | 24 |

The 1,024-command fixture uses **98.8% fewer allocations** with identical allocated bytes. Tiny and intermediate fixtures pay 8/112 extra bytes from allocator size classes and chunk boundaries; those increases are recorded rather than hidden. Median timings were 311.2 -> 284.0 ns, 15.66 -> 14.57 us and 98.03 -> 79.49 us. Raw samples are noisy, and these synthetic pruning times do not establish native latency.

### Complete tiny mock LSD row

| Path | Before median B/op | After median B/op | Before median allocations | After median allocations |
| --- | ---: | ---: | ---: | ---: |
| Public | 1,719,206 | 1,718,373 | 8,836 | 6,595 |
| Private | 1,885,479 | 1,884,604 | 8,902 | 6,641 |

Private complete-row allocations fall **25.4%**, by 2,261 allocations. Total bytes are nearly unchanged. Public timing medians were 1.147 / 1.102 ms, ranges 1.127--1.209 / 1.078--1.148 ms. Private medians were 1.199 / 1.187 ms, ranges 1.196--1.211 / 1.174--1.212 ms. Overlap limits latency conclusions. The fixture includes graph construction, admission, pruning, interpretation, downloads and accumulation, not device execution.

### Attribution and retention

Separate 20-row `-memprofilerate=1` attribution runs show `PruneTape` cumulative allocations **37,769 -> 880 objects**, about 97.7% less. Chunk allocations move into the arena helper; this cumulative comparison includes them. Flat pruning allocations were 37,593 / 308 objects.

Post-GC total in-use profiles measured 48.25 / 44.24 KiB, with no observed retained increase in this fixture. These totals include setup and test machinery and do not establish peak RSS, released-size residency or native allocator peaks. There is no mutable shared cache.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-prunearena-20261002`.
Gate script: `/workspace/tmp/pocket-prunearena-gate.sh`.
Final window: **2026-10-02 09:00:17--09:01:34 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 240 top-level + 300 subtests | 0 |
| Selected tests, race detector | 24 top-level + 30 subtests | 0 |
| Backend and affected CPU/mock regressions | 623 top-level + 1,178 subtests | 1 |

There are zero failures. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun and is not a passed gate. Its restrictions remain unchanged.

Tests cover chunk boundaries, exact final chunks, full-slice capacities, writes and appends, independent prune calls, concurrent mutations and deliberately borrowed constant values. Existing malformed/dead-source/ordering/root/budget tests, constant-only nil-command behaviour, independent parameter gradients and bound/unbound failure retention pass. Independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step references pass unchanged tolerances, as do objective endpoints, padding, all gradients/P/M/V/EMA, clipping, checkpoint/resume and tape/download accounting.

The arena helper has **100% scoped statement coverage**; `PruneTape` has **98.1%**. Its final-validation error return remains unexercised after a valid closure is constructed. Package coverage is 87.0% for `backends/training`, 68.5% for Pocket TTS and 14.4% for pockettrain under this selector. Vet includes all three affected packages; host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. No kernel or replay scope changed.

### Review and limits

A mathematical judge found the disjoint full-slice ownership contract sound and highlighted backing-allocation retention rather than logical slice length. The implementation bounds chunks at 256 integers for admitted sources, and the report records that tradeoff. The judge did not inspect source or run native code. Independent source approval, native resources/performance and voice/listening acceptance are open. Holds are unchanged.

## Next work

Initial flow tensor/operand metadata and selected-result augmentation remain host costs. Audio-hidden transformer/flow handoffs, prediction admission, dense embedding VJP, EOS scalar/diagnostic synchronisation and full snapshots/live copies still require implementation work. Continue bounded offline changes before authorised native qualification.
