# Pocket TTS shared device zero constants

## Status

Implemented separate `ptx-resident-zeroconst-hybrid-v1` and `vulkan-resident-zeroconst-hybrid-v1` scopes. They extend statecheck/lifetime execution by sharing pinned read-only positive-zero device constants of equal extent. Existing scopes retain their allocation behavior. Writable intermediates, signed-zero semantics, parameter identity, source admission and arithmetic are unchanged.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. Eight staged inference paths / 469 insertions are unchanged. Both GPU backends were disabled for deterministic offline validation. No native GPU, model, recovery, listening or publication run occurred.

## Contract

- `PlanZeroTapeStorage` performs full tape validation and existing reshape/lifetime planning, then shares equal-element constant roots only when every value is positive zero.
- Negative zero is excluded explicitly; no bit-equivalence or generated-signal hash gate establishes numerical acceptance. This admission protects the sign of zero rather than accepting changed arithmetic.
- Bound parameter IDs are protected, including zero-valued parameters. They remain independent resident allocations and never substitute for utility zero buffers.
- Shared constant slots stay pinned through tape completion, including constant outputs/reshape families. Writable outputs never share them. Native read-only operands may alias; existing stage/kernel writable-disjoint checks still apply.
- PTX allocates/uploads each unique zero slot once. Vulkan creates shape-preserving views into one constant arena per extent. Each unique owner is closed once; uncertainty retains it and forbids retry/ordinary close.
- `ResidentZeroTapeDriver` is explicit; preparation rejects missing support before allocation. Full logical source bytes still charge every tensor before pruning/storage planning. Dead-output guards, gradient roots, conservative ceilings and replay/runtime identity remain enforced.

Host zero-source pooling is separate: public host ownership remains unchanged. This slice reduces planned device constant allocations/uploads within one tape, not across tapes.

## Accounting

Tiny statecheck versus zeroconst completed-step host/mock fixture:

| Summed per-tape measure | statecheck | zeroconst |
| --- | ---: | ---: |
| Conservative source bytes | 144,560 | 144,560 |
| Lifetime-planned storage including borrowed parameters | 63,776 | 55,244 |
| Planned owned storage excluding borrowed parameters | 43,488 | 34,956 |
| Inter-tape output downloads | 116 | 116 |
| Resident diagnostics + snapshot downloads | 238 | 238 |

Graph command/tensor counts, parameter accumulation and full snapshot transfers are unchanged. These are summed logical planning values, not native peak memory, allocator overhead or released-size resource qualification.

Primitive backend mocks with two equal-sized zero constants require one constant upload instead of two and three unique allocations instead of four lifetime allocations. A zero-valued borrowed parameter remains separate from the utility zero source. Legacy tape constructors still allocate six independent owners in that chain fixture.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-zeroconst-20261002`.
Gate window: **2026-10-02 04:27:20–04:28:31 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 230 top-level + 110 subtests |
| Same cases with race detector | 23 top-level + 11 subtests |
| Affected host/mock regressions | 210 top-level + 267 subtests |

Final logs have zero failures/skips. Independent chronological SSA reference tests cover 100 deterministic random graphs with shared zero constants and NaN-prefilled mutable slots. Native-adapter mocks verify allocation/upload counts, read-only sharing, writable disjointness, protected parameter identity, signed-zero output preservation, constant-output roots, reshape retention, 65-command Vulkan plan chunks, construction/upload/launch/submit retention and blocked retry.

Full-step objective endpoints, padding, all-gradient/update/EMA parity, clipped multistep/resume, replay/scope rejection, seed/non-accumulation and unchanged transfer accounting pass. Existing numerical tolerances remain unchanged. Backend tests use mocked CUDA/Vulkan entrypoints, not native shader execution.

Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed; prior offline compiler gates were not repeated. Function coverage is recorded in `coverage-functions.txt`; it is host/mock coverage only.

### Review limits

The self-contained design judge timed out. No independent source approval or native numerical/resource/performance/voice qualification is claimed. Native read-only alias/overwrite behavior, signed-zero/FTZ policy, sustained operation and released-size resources remain unqualified pending coordinated release of holds.

## Remaining work

Graph metadata/nonzero constants still materialise before pruning. Dense embedding VJP, broader boundary fusion, cross-tape reuse, inter-tape transfers, diagnostic synchronization and full snapshots remain. No measured native speedup or complete-goal acceptance follows from these gates.
