# Pocket TTS fused shifted-audio projection

## Status

Implemented separate `ptx-resident-audiofused-hybrid-v1` and `vulkan-resident-audiofused-hybrid-v1` scopes. They extend eosfused with bound BOS/shifted-audio/input-projection tapes, removing audio assembly/projection boundaries in forward and backward. Legacy scopes and standalone builder ownership remain unchanged.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. Eight staged inference paths / 469 insertions remain unchanged. Both GPU backends were disabled. No model, native GPU, recovery, listening or publication run occurred.

## Contract

The tape forms `[BOS, normalized[:-1]]`, projects it through `flow_lm.input_linear.weight`, and computes affine hidden-input/weight VJPs. Reverse slices the first input row into the BOS gradient and pads the remaining input VJP into normalized-latent shape; the last normalized row is unused and always zero.

Forward requests only projected audio when seeds are zero, omitting reverse execution. Backward accumulates BOS and input-weight gradients once and requests only normalized-latent VJP. It does not download projected output or shifted inputs. Seeded non-accumulating calls retain their VJP; accumulating zero-seed calls retain both parameter-gradient roots.

FlowLM captures the explicit audio-fusion interface at forward. It no longer builds/stores the shifted host audio input in this scope. Backward uses the fused interface and rejects its removal or disabled scope. Voice projection, sequence ordering, transformer semantics and objective scaling remain unchanged. EOS fusion and prior residency/lifetime/zero/state-check behavior remain inherited.

Full input/topology/finite/source-byte/binding admission occurs before native execution. Public builders own inputs/parameters; private resident construction borrows parameter values read-only as before. No new kernel or arithmetic reassociation was introduced. Uncertain failures remain terminal with owners retained.

## Accounting

Tiny eosfused versus audiofused complete-step host/mock fixture:

| Measure | eosfused | audiofused |
| --- | ---: | ---: |
| Tapes | 47 | 45 |
| Boundary tensor downloads | 113 | 109 |
| Commands | 3,959 | 3,962 |
| Conservative logical tape bytes | 144,432 | 144,424 |
| Summed planned owned tape bytes | 34,824 | 34,664 |

BOS/input parameter-accumulation counts, resident diagnostic syncs and full snapshots remain unchanged. Added shape/slice commands connect components without changing their arithmetic. Counts are offline logical/mock evidence, not native peak memory or latency measurements.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-audiofused-20261002`.
Gate window: **2026-10-02 04:45:44–04:46:30 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 190 top-level + 170 subtests |
| Same cases with race detector | 19 top-level + 17 subtests |
| Affected host/mock regressions | 175 top-level + 244 subtests |

Final logs contain zero failures/skips. Independent scalar F64 affine/input/weight/BOS VJP references cover 1/2/7/257 frames. Independently composed original assembly/projection/assembly-VJP paths match existing tolerances. Cases cover final-row zero VJP, public ownership, mask-independent input/seed topology, finite/shape/budget admission, unsupported bias and borrowed-output rejection.

New-scope full-step objective endpoints, padding, gradients/updates/M/V/EMA, clipped multistep/resume, checkpoint/replay isolation, failures, zero/seeded accumulation and owner/scope guards pass. Tests verify that fused FlowLM tapes do not materialise host shifted input. Legacy FlowLM/sequence/projection/EOS and CPU/PyTorch full-step regressions remain passing; no numerical tolerance changed.

Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed; prior offline compiler gates were not repeated. Coverage details are in `coverage-functions.txt`; coverage is host/mock only.

### Corrections and review limits

Initial full-step download expectations assumed the previous audio assembly/projection path; updated new-scope accounting records the four removed downloads without relaxing numerical gates. Initial and corrected logs are retained.

The self-contained design judge timed out. No independent source approval or native numerical/resource/performance/voice acceptance is claimed. Device behavior, strict-FP/FTZ, released-size resources and sustained quality remain unqualified pending coordinated release of holds.

## Remaining work

Voice projection, sequence/transformer/flow boundaries, host graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronization and full snapshots remain. This is a further transfer-inclusive hybrid slice, not full-device training or complete-goal acceptance.
