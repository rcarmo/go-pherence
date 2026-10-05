# Pocket TTS resident weighting sample packing

## Status

Implemented separate `ptx-resident-sampleweight-hybrid-v1` and `vulkan-resident-sampleweight-hybrid-v1` scopes. They extend devicescale's resident graph with sample-column and logvar-seed packing inside the weighting MLP tape. This is a bounded device-completeness slice, **not** full-device training or a demonstrated performance improvement.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes remain uncommitted. The eight staged inference paths / 469 insertions are unchanged. Both GPU backends were disabled for deterministic offline validation. No GPU probe, model run, listening, recovery or publication occurred. Native qualification and existing holds remain open.

## Contract

- `WeightSampleInputs` supplies three frozen time columns (`diagonal`, `s`, `t`) and two already-scaled logvar-seed columns. Each contains F entries, for 1 <= F <= 512.
- The weighting tape forms `[2F,2]`: first F rows `[diagonal[i],diagonal[i]]`, then F rows `[s[i],t[i]]`. The diagonal column is uploaded once and referenced twice. Seeds form `[2F,1]` in the same branch order.
- Existing row scatter, vector pad/sum and reshape commands perform packing. No new shader/PTX entry or kernel compilation is required. The affine/ReLU MLP and its VJP retain their arithmetic and parameter bindings.
- Loss seeds have already received objective/valid-row scaling. Packing applies **no additional scale**. One seeded weighting call accumulates parameter gradients; the preceding forward call does not.
- Returned `DInput` remains the MLP input VJP in `[2F,2]` order, not a derivative propagated through packing into frozen time columns. Endpoint input-only VJP semantics are unchanged.
- Source graph, parameter inventory, finite columns/topology and byte ceilings validate before materialisation/pruning. Non-accumulating zero-seed forward execution omits reverse roots; nonzero-seed input VJPs and zero-seed accumulating parameter gradients are retained.
- The complete step admits active-row timing/mask constraints as before; the builder allows finite padding times outside active timing ranges, preserving the existing padding contract.
- Only the new scope uses the optional sample-weight interface. Legacy weighting scopes still pack on CPU; the legacy builder preserves its tensor/command order. CPU defaults, old replay identities and numerical tolerances remain unchanged.
- Admission/execution failures poison the active graph and retain uncertain owners. Shared weighting/component owner checks and independent replay identity still apply.

## Accounting

| Tiny completed-step host/mock accounting | devicescale | sampleweight |
| --- | ---: | ---: |
| Commands | 3,891 | 3,958 |
| Tensor specs | 4,996 | 5,078 |
| Summed logical tape bytes | 142,880 | 144,560 |
| Boundary tensor downloads | 116 | 116 |

Tape/parameter-accumulation counts remain unchanged. Host duplicate-time packing and seed concatenation are no longer consumed. Raw time/seed constants total 5F floats rather than the legacy packed 6F on a seeded weighting call; additional placeholder/packing tensors also exist. This does **not** establish lower total uploaded bytes, native peak memory or latency. Graph/constants still materialise before pruning; per-tape intermediates, inter-tape transfers and full snapshots remain.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-sampleweight-20261002`.
Gate window: **2026-10-02 02:11:07–02:11:50 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 170 top-level + 110 subtests |
| Same cases with race detector | 17 top-level + 11 subtests |
| Affected host/mock regressions | 99 top-level + 107 subtests |

Zero failures/skips in final logs. Independent scalar F64 affine/ReLU forward/VJP references cover F=1/3/257/512; a separately host-packed legacy path checks output/input/parameter-gradient parity. Exact ReLU-zero cases preserve the nondifferentiable-boundary convention. Numerical tolerances remain `2e-7 + 2e-6*abs(reference)` for the independent oracle and the existing pinned full-step/update tolerances.

Cases include full-step objective endpoints, padding, clipped multistep M/V/parameter/EMA parity, checkpoint/resume, scope rejection, owner/shape/budget/nonfinite-column admission, owned outputs, forward pruning, seeded non-accumulating input VJP, zero-seed accumulation and failure retention. A test poisons host `weightInputs`/`dLogvars` scratch after the first tape while preserving resident ownership: the remaining step succeeds and leaves those arrays unconsumed/unrebuilt.

Coverage for the new file: validation, graph packing, builder and scope predicate 100%; resident evaluator 95%. This is host/mock coverage, not native coverage. Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, layout checks and diff checks passed. Existing kernel compilation gates were not repeated because this slice adds no kernel.

### Corrections and review limits

Initial development logs preserve corrected helper-name/import errors, incorrect vector-pad argument order and row-placeholder extent. A forwarding-owner scratch test was replaced with a test-driver callback, preserving the real ownership guard rather than weakening it.

An initial F64 oracle case sat at a ReLU rounding boundary; F32 and F64 selected different subgradients. Reference samples were moved away from that nondifferentiable point without changing tolerances or implementation arithmetic. Separate exact-zero ReLU cases remain in the final gate. Failed and corrected logs are retained.

The independent delegate design review timed out. No independent source approval or native arithmetic approval is claimed. Offline references and source inspection do not establish device FTZ/trig/strict-FP behavior, released-size resources, sustained operation, speed or voice quality.

## Next

Broader boundary fusion, avoiding host graph/constants construction before pruning, dense embedding VJP/live allocation reductions, fewer inter-tape transfers/diagnostic synchronization and smaller completed-step snapshots remain. Native numerical/resource/performance qualification and new-voice/listening acceptance require coordinated release of the holds.
