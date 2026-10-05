# Pocket TTS resident trainer-load storage reuse

## Status

Completed resident updates now reuse valid live moment/EMA arrays through a private loader. Public `LoadState` restore behavior, backend boundary ownership, replay arithmetic and full device snapshot transfers are unchanged.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes remain uncommitted. The eight staged inference paths / 469 insertions are unchanged. Validation was deterministic and offline with both GPUs disabled. No model, GPU probe, recovery, listening or publication occurred.

## Contract

`loadResidentCompletedState` accepts only the independent owned completed snapshot used by resident graph/optimiser backends. It is not an arbitrary checkpoint restore API.

- Header admission requires version 1, unchanged valid config/EMA policy and exactly the next completed step.
- Read-only validators admit complete parameter, buffer, moment and enabled EMA inventory/shape/finiteness before writes. Source and live V must be nonnegative. Empty live parameter/buffer storage is rejected.
- Existing live moment/EMA arrays are reused only with valid inventory, lengths and finite values. Missing initial moments allocate once; missing completed moments or enabled EMA are rejected.
- A sorted address-range admission rejects overlap across all source/destination parameter, frozen buffer, M/V and EMA arrays. This prevents cross-tensor copy corruption or aliases between the owned backend boundary and mutable trainer storage. No fallible work remains after admission.
- Copying populates existing live arrays, staged maps are published, then step count advances. Rejection performs no partial value writes. The trainer remains single-owner/non-concurrent.
- Public `LoadState` still replaces moments/EMA with independently owned restored arrays. Private rejection after device update is terminal through the existing backend failure path; it never retries uncertain updates.

This eliminates transient public-load clones and repeated live moment/EMA allocation. Copies from the owned snapshot into live state still occur; device-to-host snapshot bytes and per-step boundary transfers remain unchanged.

## Evidence

Directory: `/workspace/exports/pocket-gpu-training-loadreuse-20261002`.
Final main gate: **2026-10-02 02:38:26–02:39:10 UTC**; empty-buffer guard follow-up followed it.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 40 top-level + 350 subtests |
| Same cases with race detector | 4 top-level + 35 subtests |
| Affected host/mock regressions | 116 top-level + 175 subtests |
| Final guard admission follow-up (`-count=10`) | 10 top-level + 300 subtests |
| Follow-up race | 1 top-level + 30 subtests |

Final logs contain zero failures/skips. Tests cover multistep parity with public loading, EMA enabled/disabled, source/live independence, stable live M/V/EMA backing arrays, complete admission before writes, malformed headers/inventory/shapes/nonfinite values, negative V, source-source/source-live/live-live aliases and address-range guards. Four PTX/Vulkan graph/optimiser runner scopes preserve live arrays through updates and checkpoint successfully. Existing resume, full-step/AdamW/EMA, boundary mutation, owner retention and atomic checkpoint restore tests pass.

Main targeted coverage: loader 97.6%, range validator 100%; the final empty-buffer guards have separate repeated/race evidence. Affected-package vet, host build, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed; vet was repeated after the small final guards. No kernel changed, so compiler gates were not repeated.

### Allocation and timing

Synthetic weighting topology includes a 256-by-256 middle matrix. Three 100-iteration steady-state measurements:

| Isolated load | Bytes/op | Allocations/op | Observed time |
| --- | ---: | ---: | ---: |
| Public replacement | 1,128,096–1,128,127 | 276 | 697–746 µs |
| Private reuse | 47,509–47,515 | 30 | 717–836 µs |

This establishes allocation reduction, **not** a latency improvement. Timings overlap and one reuse run is slower. The first implementation scanned source arrays twice and measured roughly 1.01 ms versus 0.68–0.76 ms public loading. Duplicate scans were removed without dropping inventory/finite/header/alias admission, and the full final gates were rerun. Initial and single-pass logs are retained. The benchmarks exclude device execution, snapshot download and complete training requests.

No hash or generated-signal equality gate was used. Exact state-bit comparisons in rejection tests only verify that malformed/nonfinite owned state was not mutated, including NaN payloads; numerical parity retains existing tolerances.

### Review limits

The self-contained design judge timed out. No independent source approval or native resource/numerical/performance/voice acceptance is claimed. Offline source inspection and host/mock tests do not establish device behavior.

## Remaining work

Full snapshots still download each update; owned backend arrays and trainer arrays remain separate. Graph/constants materialisation before pruning, dense embedding VJP/live device allocations, broader boundary fusion, inter-tape transfers and diagnostic synchronization remain. Native resources/performance and new-voice/listening qualification require coordinated release of existing holds.
