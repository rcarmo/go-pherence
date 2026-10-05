# Pocket TTS graph-local immutable zero sources

## Status

Private synchronous resident tape builders now share literal-zero host source arrays within one graph, keyed by exact element count. Public builders retain independently owned arrays. Tensor IDs, shapes, commands, arithmetic, native allocations/transfers and logical byte admission are unchanged. No new replay scope is required because only read-only host construction storage changes.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training work is uncommitted. Eight staged inference paths / 469 insertions are unchanged. Validation was offline with both GPU backends disabled. No model, GPU probe, recovery, listening or publication run occurred.

## Ownership

`trainingZeroPool` belongs to one private flow or transformer-style graph. Each literal zero tensor still receives a distinct ID and shape; matching extents share only immutable host upload-source storage. Nonzero constants, caller-provided zero inputs/seeds and parameters are not inserted into this pool. Parameter borrowing retains its separate read-only contract.

Transformer, weighting, projection and sequence materialisation use the pool only after full source-byte admission. The flow builder avoids the prior make-plus-copy zero helper for private resident construction. Public builders retain the original ownership path.

The pool has no global/cross-graph cache. Source arrays stay reachable through tensor specifications. Initial-value tensors remain forbidden as command destinations, so private zero arrays never become execution outputs. Native drivers still allocate/upload each constant tensor independently; source pooling does not change device alias rules, logical ceilings, lifetime allocation, pruning or failure retention.

## Evidence

Directory: `/workspace/exports/pocket-gpu-training-zeropool-20261002`.
Gate window: **2026-10-02 04:17:59–04:18:38 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 40 top-level + 80 subtests |
| Same cases with race detector | 4 top-level + 8 subtests |
| Affected host/mock regressions | 161 top-level + 219 subtests |

Final logs have zero failures/skips. New cases prove equal-extent reuse within a graph, different-extent/cross-graph separation, public independent storage, caller input/seed ownership even when values are zero, immutable-source admission and unchanged logical byte ceilings. Eight independent graph subtests exercise local construction under the race detector. Existing component builder topology/constants/VJP, pinned full-step/update/EMA, objective endpoint, padding, resume, scope/admission and failure-retention gates remain passing without changed tolerances.

Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No device kernel changed; previous offline compiler gates were not repeated.

### Allocation fixture

Isolated materialisation of 64 zero arrays, each 32,768 F32 elements, three 100-iteration measurements:

| Host source storage | Bytes/op | Allocations/op |
| --- | ---: | ---: |
| Independent arrays | 8,388,611–8,388,664 | 64 |
| One graph-local extent pool | 131,408 | 3 |

This intentionally isolates repeated zero-source allocation. It is not a measured complete-request allocation, native peak memory, device-transfer reduction or throughput result. Graph metadata and other constants still materialise before pruning.

### Corrections and review limits

The initial concurrent test inspected a caller seed tensor rather than the literal tangent-zero tensor. It now derives the zero tensor from the forward command's tangent operand, avoiding fixture-specific IDs; failed and corrected logs are retained. A development edit initially matched both field and function-parameter text and was narrowed before application.

The self-contained design judge timed out. No independent source approval or native arithmetic/resource/performance/voice qualification is claimed. Private pooled arrays are read-only by contract; concurrent mutation of a live graph/model remains unsupported.

## Remaining work

Host graph metadata/nonzero constants, dense embedding VJP, broader boundary fusion, inter-tape transfers, diagnostic synchronization and full snapshots remain. Native resources/performance and new-voice/listening qualification require coordinated release of existing holds.
