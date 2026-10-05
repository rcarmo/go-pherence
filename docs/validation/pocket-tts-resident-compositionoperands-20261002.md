## Pocket TTS composition operand storage

Composition remapping now uses bounded operand-copy storage instead of allocating one input list per remapped command. The two profiled loops account for most remaining composition objects after the flow-command arena change. Reusing the existing helper reduces private complete-row allocations by **41.9%**, with a **0.08%** allocated-byte cost.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. All measurements and gates used offline CPU/mock execution with NVIDIA/Vulkan disabled. No native GPU, released model/audio, recovery or publication work occurred.

## What changed

`composeDistillTrainingTapes` and `appendAdmittedFlowGraph` each create a local `flowOperandArena` before remapping commands. Each input list is copied into a disjoint region, then the existing mapping is applied. Output remapping and command append order are unchanged. Separate append invocations have separate arenas; the primary and endpoint graphs do not share mutable remapped operand storage.

The helper already used by flow graph emission returns nil for empty lists, capacity-limited independent slices for larger lists, and disjoint capacity-limited regions of 256-integer chunks for supported arities up to eight. Retaining a bounded operand slice can pin at most 2 KiB on amd64. No global cache or inter-call mutable storage was added. Chunk padding explains the small allocated-byte increase; logical tensor bytes and their admission ceilings do not include host metadata, as before.

Complete source tape and resident binding admission still precedes remapping. Combined ceilings and binding admission still follow composition, before pruning. IDs, modes, offsets, scales, operand order, tensor values, output roots, primary-only gradient roots and parameter scaling remain unchanged. Endpoint stop-gradient and exactly-once accumulation retain their existing contracts. No replay/backend scope or numerical tolerance changed.

This change owns only the operand lists that were previously cloned for remapping. Shapes, constants and unremapped leading-component command arrays keep their existing read-only borrowing contract. It does not promise independence for arbitrary mutation of those borrowed source arrays. New tests mutate only the newly owned regions.

## Complete-row measurements

Before and after test binaries ran alternately for five samples per path, each measuring 100 warmed tiny CPU/mock rows. Fixture preparation was outside measurement. Both binaries used the operand-arena baseline; only composition remap storage differs. Offline Go, `GOMAXPROCS=2`, `-p=1 -parallel=1` and both GPU disable flags were fixed.

| Path | Before median B/op | After median B/op | Before median allocations/op | After median allocations/op |
| --- | ---: | ---: | ---: | ---: |
| Public tiny row | 1,742,331 | 1,743,879 | 3,022 | 1,735 |
| Private tiny row | 1,908,563 | 1,910,106 | 3,068 | 1,781 |

Both paths save 1,287 allocations per row. Private allocation count falls 41.9%; allocated bytes rise about 1.5 KB/op (0.08%). Public allocation count falls 42.6% with the same small byte cost. These are host allocation reductions, not GPU memory or native throughput results.

Public timing medians were 1.045 / 1.028 ms, with ranges 1.028--1.147 / 1.011--1.057 ms. Private medians were 1.171 / 1.114 ms, with ranges 1.126--1.221 / 1.098--1.159 ms. The ranges overlap. No native speedup follows from these tiny mock timings.

Separate bounded 20-row attribution runs with `-memprofilerate=1` allocated **63,983 -> 42,448 objects**, about 33.7% less including setup. `appendAdmittedFlowGraph` flat attribution falls 14,872 -> 484 objects, with another 220 objects attributed to its operand helper. `composeDistillTrainingTapes` falls 7,788 -> 308 flat objects. The old per-command clone sites disappear; chunk objects are attributed to `flowOperandArena.copy`.

Post-GC total in-use profiles measured **43.35 / 43.24 KiB**, essentially unchanged. Those totals include test/regexp setup; they do not prove released-size residency, peak RSS or native allocator peaks. The mock interpreter is now the largest flat object contributor and is not a production GPU optimisation target.

## Offline gates

Evidence directory: `/workspace/exports/pocket-gpu-training-compositionoperands-20261002`.
Gate script: `/workspace/tmp/pocket-compositionoperands-gate.sh`.
Window: **2026-10-02 09:23:40--09:24:47 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 250 top-level + 320 subtests | 0 |
| Selected tests, race detector | 25 top-level + 32 subtests | 0 |
| Affected CPU/mock regressions | 632 top-level + 1,178 subtests | 1 |

Zero failures. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

New tests reconstruct and check the complete remap contract (IDs, command order, modes and scales), then exercise source/sibling write isolation, append isolation, primary/endpoint append independence, retained numerical outputs, separate compositions and concurrent compositions from the same immutable sources. Existing tests verify malformed complete sources, binding/seed/input extents, graph/byte ceilings and source preservation across success and late failure.

Both distillation composition source files and the operand-copy helper retain **100% scoped statement coverage**. Package coverage under the regression selector is 87.0% for `backends/training`, 68.6% for Pocket TTS and 14.4% for pockettrain. Independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step references, scaling endpoints, padding, gradients/updates/EMA, replay/checkpoint/resume, retention and accounting tests pass unchanged tolerances.

Vet covers all three affected packages. Host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass. The diagnostics tool has no Go validator; the Go tests, vet and builds supply the checks. No kernels changed.

## Review limits and next work

The delegated source review timed out after 60 seconds, so no independent source approval is claimed. A separate bounded invariant judge agreed that copying before mapping, non-overlapping advancing chunk windows, capacity limits and invocation-local arenas preserve immutable sources and owned remapped slices. It did not inspect source or execute native code. Native accuracy/resources/performance and voice/listening acceptance remain open.

Remaining production allocation contributors include flow AD node metadata and complete-source/binding admission. Transformer/flow audio-hidden handoffs, forward prediction admission, dense embedding VJP, EOS diagnostics and full snapshots still need implementation work. No hold is released by this offline slice.
