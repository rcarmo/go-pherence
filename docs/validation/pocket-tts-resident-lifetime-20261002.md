# Pocket TTS intra-tape lifetime storage

## Status

Implemented separate `ptx-resident-lifetime-hybrid-v1` and `vulkan-resident-lifetime-hybrid-v1` scopes. They extend sampleweight with explicit intra-tape intermediate allocation reuse. Legacy constructors/scopes keep independent storage; arithmetic and graph order are unchanged.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. The eight staged inference paths / 469 insertions remain unchanged. Both GPUs were disabled throughout deterministic offline validation. No native GPU, model, recovery, listening or publication run occurred.

## Storage contract

- `PlanTapeStorage` validates the complete supplied tape, canonicalises reshape families and computes first-write/last-consumer intervals. A declared output pins its entire reshape family through terminal download/gradient accumulation.
- Constants and unwritten/dead tensors remain pinned. Constants upload before execution, so even an early last read does not make them reusable. Borrowed resident parameters remain separate owned device allocations.
- Only equal-element intermediate allocations can share a slot, and only when the previous last consumer is **strictly before** the next write. A dispatch output never aliases its inputs. No cross-tape reuse, operation folding, reassociation or implicit pruning is added.
- PTX retains each unique allocation once and synchronises each command. Vulkan allocates one arena per unique slot, reshapes views for rank metadata, and retains its existing compute-stage read/write barriers. Ordered plan chunks retain submission/fence ordering; a 65-command mock covers the 64-stage boundary.
- Lifetime-mode buffer access/downloads accept only declared outputs. Dead intermediates cannot escape after their storage is reused. Legacy output access remains unchanged.
- `ResidentReuseTapeDriver` is explicit; preparation rejects missing support before native allocations. The resident coordinator still validates the complete source/binding inventory and logical byte ceiling before pruning, retains requested outputs/all accumulated gradients, and blocks updates/snapshots during borrowing.
- Attempted native execution failure retains every unique owner and blocks retry/ordinary close. Construction cleanup and borrowed-parameter ownership remain unchanged.

No new kernel or weakened byte ceiling was introduced. This remains transfer-inclusive hybrid training, not full-device execution or native resource qualification.

## Accounting

Tiny sampleweight versus lifetime completed-step host/mock fixture:

| Summed per-tape measure | sampleweight | lifetime |
| --- | ---: | ---: |
| Commands | 3,958 | 3,958 |
| Tensor specs | 5,078 | 5,078 |
| Conservative logical source bytes | 144,560 | 144,560 |
| Storage accounting with reshape aliases | 93,368 | 63,776 |
| Owned storage excluding borrowed parameters | 73,080 | 43,488 |
| Boundary tensor downloads | 116 | 116 |

The 29,592-byte reduction isolates reuse from pre-existing reshape aliases and borrowed parameters. These are sums across tapes, **not native peak memory**, allocator/plan overhead or released-size resource measurements. The primitive four-command backend mocks reduce six independent allocations to four unique owners, or three owned allocations with one borrowed parameter.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-lifetime-20261002`.
Main gate: **2026-10-02 03:58:57–04:00:07 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 230 top-level + 110 subtests |
| Same cases with race detector | 23 top-level + 11 subtests |
| Affected host/mock regressions | 185 top-level + 245 subtests |
| Refined accounting follow-up | 10 top-level; 1 race |

Final logs have zero failures/skips. Tests include 100 deterministic random SSA graphs evaluated chronologically against separate scalar storage with NaN-prefilled pool slots; reshape chains, output-family retention, unequal extents, constant-only/dead storage, full source ceiling/dependency admission, missing driver capability and dispatch disjointness. Existing numerical tolerances were not changed.

Backend mocks verify allocation counts, PTX arithmetic against the independent chain result, borrowed parameter pinning, legacy allocation behavior, declared-output guards, unique closes, construction rollback and attempted launch/submit failure retention. Vulkan mocks cover stage barriers and multi-plan execution; they do not execute native shaders.

Full-step host pooled execution covers all-gradient/update/EMA parity, objective endpoints, padding, clipped multistep/resume, replay/scope rejection, seeded non-accumulating VJP and zero-seed parameter accumulation. Arithmetic is compared with the existing pinned/reference paths, not hashes.

`PlanTapeStorage` targeted coverage is 100%; adapter/constructor coverage is recorded in `coverage-functions.txt`. The shared declared-output helper is exercised from backend tests, but its own package coverage report does not include cross-package instrumentation. Coverage is host/mock only.

Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed; prior compiler gates were not repeated.

### Corrections and review limits

The first pooled host-interpreter run failed because its reduction implementation used `+=` into fresh zero arrays. Native kernels overwrite their complete destinations. The interpreter now clears reused non-reshape destinations to emulate that overwrite contract; no production arithmetic or tolerance changed. Failed and corrected logs are retained. Initial mock signature/type/extent errors were corrected before final gates.

The self-contained design judge timed out. No independent source approval, native numerical/resource/performance acceptance or voice-quality qualification is claimed. Native overwrite coverage, barriers/device behavior, released-size resources, FTZ/strict-FP and sustained operation remain unqualified until coordinated release of holds.

## Remaining work

Graph metadata and nonparameter constants still materialise before pruning. Dense embedding VJP, broader boundary fusion, inter-tape transfers, diagnostic synchronization and full snapshots remain. No measured native speedup or complete-goal acceptance follows from these source/offline gates.
