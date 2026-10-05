# Pocket TTS device-merged post-update checks

## Status

Implemented separate `ptx-resident-statecheck-hybrid-v1` and `vulkan-resident-statecheck-hybrid-v1` scopes. They extend lifetime mode with one device-merged post-update finite/nonnegative diagnostic download. Optimiser arithmetic, full snapshot transfers and legacy per-parameter validation paths are unchanged.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes remain uncommitted. The eight staged inference paths / 469 insertions are unchanged. Both GPU backends were disabled for deterministic offline validation. No native GPU, model, recovery, listening or publication run occurred.

## Implementation

`NewStateCheckResidentState` enables global norm, direct device-scale AdamW and merged mutable-state validation. It reuses the existing global A/B four-scalar buffers after gradient clipping; resident logical storage remains the existing additional 48 bytes.

Before reduction or mutation, complete metadata/kernel admission validates every post-update merge range. The constructor bounds total mutable-state elements, counting parameter/M/V/EMA separately, at `1<<38` (parameter total <= `1<<36`). Conservative source/resident byte ceilings are unchanged.

After AdamW, the coordinator resets the diagnostic summary, visits named parameters in deterministic order and computes existing chunk `NormStats` for parameter, M, V and EMA. V alone requires nonnegative admission; enabled and disabled EMA storage are both checked as before. Existing global maximum-scan mode 0 validates each chunk summary and carries its invalid flag. No square accumulation, clipping decision or new kernel is required for these checks.

One four-scalar result downloads. Admission requires finite nonnegative maximum, zero sum, zero invalid flag and zero correction. Any upload/statistics/merge/download/invalid-result error after mutation is terminal: step count is not committed, and retry/snapshot/ordinary close remain blocked with owners retained. Success still waits for all checks before replay commit.

Legacy states retain their four per-parameter chunk-summary downloads and host validation. No existing replay identity or numerical tolerance changed.

## Transfer accounting

The 59-parameter tiny completed-step host fixture retains its 116 inter-tape output downloads and four snapshot arrays per parameter:

| Resident download calls | lifetime | statecheck |
| --- | ---: | ---: |
| Gradient norm diagnostic | 1 | 1 |
| Post-update diagnostic | 236 | 1 |
| Full parameter/M/V/EMA snapshots | 236 | 236 |
| Total resident diagnostic + snapshot | 473 | 238 |

Thus update diagnostics fall **237→2 calls**, while snapshots are unchanged. Tape commands/tensors/lifetime storage and parameter-gradient accumulation remain unchanged. The post-update path adds device merge dispatches and one small reset upload; fewer transfers do not establish lower native latency or total request speed.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-statecheck-20261002`.
Gate window: **2026-10-02 04:09:53–04:11:04 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 220 top-level + 260 subtests |
| Same cases with race detector | 22 top-level + 26 subtests |
| Affected host/mock regressions | 155 top-level + 263 subtests |

Final logs have zero failures/skips. New cases cover unchanged clipped multistep updates/snapshots, resume, EMA enabled/disabled, invalid parameter/M/V/EMA values, negative V, diagnostic decoding, invalid carry, 1/255/256/257/4097-element chunk boundaries, maximum/subnormal finite values and last-chunk nonfinite rejection. Pure preflight errors do not mutate or upload; post-update reset/statistics/merge/download failures retain owners without step commit or retry.

Full-step objective endpoints, padding, gradients/updates/EMA, seeded non-accumulation, checkpoint/replay scope isolation and lifetime accounting pass under the new scope. Existing globalnorm/device-scale paths and PTX/Vulkan offline merge/kernel ABI/retention tests remain passing. All four new state-check functions have 100% targeted host/mock coverage.

Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed, so prior offline compiler gates were not repeated.

### Corrections and review limits

Initial full-step tests counted only one summary reset upload; the expectation now includes the post-update reset for this scope, while still rejecting host parameter-gradient uploads. Initial and corrected logs are retained.

A self-contained design judge timed out. No independent source approval or native arithmetic/resource/performance/voice qualification is claimed. Existing native NormStats/max-scan behavior, FTZ/strict-FP and resource limits remain unqualified pending coordinated release of holds.

## Remaining work

The pre-update norm diagnostic still synchronises with host. Full snapshots, CPU dispatch/boundary assembly, graph metadata/nonparameter constants, dense embedding VJP and inter-tape transfers remain. This is a further transfer-inclusive hybrid slice, not full-device training or complete-goal acceptance.
