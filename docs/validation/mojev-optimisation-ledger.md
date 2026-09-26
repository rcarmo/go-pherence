# MoJev optimisation ledger

Keep useful local optimisations even when end-to-end timing is inconclusive.
Improvements can accumulate, and server contention can obscure small latency
gains. Numerical fidelity, ownership, bounded resources and supported fallbacks
remain acceptance gates. Statistical non-significance alone is not a rejection.

## Adopted

| Change | Code checkpoint | Evidence and limits |
|---|---|---|
| Reusable SIMD workers/scratch and assembly attention | `3860f490` | Request allocations reduced; six-worker cap retained. |
| Batched CUDA launches and nonreflective driver binding | `85fff290`–`b7fdf685` | Reduced launch/allocation overhead; no blanket latency inference. |
| Shared ancestor trees and bounded candidate groups | `f8a8a81a`–`1ea0b32f` | Branch-local isolation preserved; independent long/grouped references pass. |
| NVIDIA 32×64 F32 GEMM, 128 threads | `33bafe6d` | Measured four-workload latency improvement; [event measurements](mojev-gemm-events-20260926.md). |
| GPU host encoder release | `84061f99` | About 1.99 GB less retained heap when CPU scorer is dropped; [lifetime report](mojev-gpu-host-lifetime-20260926.md). |
| Packed-only SIMD projection storage | `d99bd9d4` | About 1.99 GB less retained heap; [storage and retention report](mojev-simd-packed-only-20260926.md). |
| Byte-symbol table and call-local BPE scratch | `060b62c0` | Tokenizer microbenchmarks 39–45% faster; full-request allocations down roughly 40–64%; [report](mojev-tokenizer-allocations-20260926.md). |
| Direct encoded segments for native scoring | `87406572` | Avoids masks/re-extraction, fewer request allocations; [report](mojev-direct-segments-20260926.md). |
| Four-row SIMD recurrence | `ca406642`, merged as `649bf072` | Adopted with measured 31.73% local speedup, exact state/output transitions and unchanged independent model parity; full-request benefit remains uncertain. Details below. |
| Owned weight transfer during loading | After `6246a7cc` | About 3.01 GB less CPU-load allocation; measured SIMD peak RSS 5.78–5.90 → 4.76 GiB. [Ownership and load report](mojev-owned-load-20260926.md). |

## Four-row recurrence

`qwen35BranchDeltaHead` groups four independent 128-element rows, sharing each
Q/K vector load through the existing amd64 `Sdotx4` kernel. Each row still does
scale → memory dot → update → output dot → scale. The amd64 implementation keeps
the existing two-accumulator dot-product pattern. Unsupported targets and
assembly-disabled dispatch use the previous row-at-a-time loop. Candidate
fork/restore logic, cancellation checks and the six-worker projection cap are
unchanged. No approximation or wider tolerance was introduced.

On i7-12700, Go 1.26.3/Linux amd64, `GOMAXPROCS=6`, ten microbenchmark samples
measured 3.360 → 2.294 µs per head update (−31.73%, p<0.001). Both paths allocate
zero bytes/objects on this host. Tests compare every state/output element
bitwise across twelve updates in native, disabled-dot and scalar modes, check
source preservation, guards and warm allocations. These tests run repeatedly
under the race detector.

Six interleaved processes per version, each with one warm-up and five calls,
measured the following full-request medians against `87406572`:

| Workload | Row-at-a-time ms | Four-row ms |
|---|---:|---:|
| Short, two choices | 243.4 | 241.2 |
| Short, eight choices | 373.8 | 352.7 |
| Two questions, two choices | 473.7 | 454.2 |
| Longer, two choices | 765.1 | 749.2 |

All 48 workload example comparisons were exact. The time geomean was 3.21%
lower, but individual comparisons were not statistically significant and the
changed-run intervals varied by as much as 51%. The old measurements did not
record host contention or throttling, so server load is a plausible explanation,
not an established cause. These results neither prove a full-inference speedup
nor justify throwing away the measured local gain.

The approved checkpoint is unchanged; independent 512-path and 4096-total-token
ordinary SIMD tests pass. Grouped errors are `1.40071e-6` logits and `5.72205e-5`
hidden. Long cases remain below `6.55651e-7` logits and `9.15527e-5` hidden,
against unchanged `3e-4` / `2e-3` gates. Two concurrent grouped requests pass the
ordinary test. The adopted four-row path also passes the released SIMD race
in 189.13 seconds, with sampled hidden error `9.91821e-5` and base-logit error
`3.69549e-6`. The merged whole-tree NVIDIA-disabled race rerun exits zero;
the first invocation was aborted and is preserved separately. Vet/build and
docs/layout checks pass. CI run `36232885986` passed for `649bf072`.

Focused review found no row mixing, ordering, shape or lifetime issue. The
preservation branch passed affected-package races three times, vet/build and
ARM64/RISC-V Qwen test-binary cross-builds; foreign binaries were not executed.
The branch `experiments/mojev-delta-four` retains the tested source separately.
Raw samples, sources, logs and benchstat are under
`/workspace/tmp/mojev-delta-four-20260926`.

## Preserved candidates and experiments

These are retained for cumulative testing; being absent from main does not mean
they are forgotten or disproven. Establish their current correctness and costs
before combining them. Do not reclassify older failures as passes.

| Candidate | Existing evidence | Revisit condition / location |
|---|---|---|
| Paired GPU MLP gate/up projection | Exact four-workload responses; mixed timing, including slower workloads | Check interaction with launch/tiling changes. `mojev-gateup-20260926/{paired.cu,nvidia-paired.go,paired.json}` under `/workspace/tmp`. |
| Warp-local GPU attention | Exact four-workload responses; small inconclusive gain | Independent long/grouped gates and load-aware paired timing before adoption. `mojev-gateup-20260926/{warp.cu,nvidia-warp.go,warp.json}`. |
| Precomputed GPU recurrence parameters | Exact four-workload responses; marginal lower medians | Test alone and combined with warp attention, retaining exact operations. `mojev-gateup-20260926/{prepared.cu,nvidia-prepared.go,prepared.json}`. |
| CPU padding/stride/prefetch/unroll/K-block variants | Mixed/no consistent request gains in prior tests | Preserve assembly and logs; revisit only as specified combinations or changed workloads. `/workspace/tmp/mojev-cpu-tiles-20260926`; prior [record](mojev-long-f32-normalisation-20260926.md). |
| NVIDIA 16×64 and 32×128 GEMM tiles | Slower in the recorded experiment | Different geometry/device may change the result; current 32×64 remains selected. [GEMM record](mojev-gemm-events-20260926.md). |

Workspace artifact paths above are evidence locations, not portable source
checkouts. The four-row source is now in Git; older workspace-only prototypes
need an audited patch before reuse. No trial replaces the numerical reference.

## Load-aware cumulative measurements

Future comparisons must retain the baseline and candidate binaries and record
both individual and combined effects. For compatible candidates A and B, use
baseline/A/B/A+B with interleaved order, followed by the same accuracy, ownership
and resource gates. Reprofile the combined implementation rather than assuming
that isolated percentage gains add arithmetically.

Record host CPU utilisation, runnable work, CPU pressure/stolen time where
available, cgroup CPU quota and throttling deltas, affinity, and competing
processes alongside timing. For GPU trials also record utilisation, clocks and
temperature. Keep collector overhead identical and label collector runs.
Do not stop unrelated services or change host governors for a benchmark without
approval. Preserve busy-host results rather than selecting only favourable
samples; distinguish contention evidence from guessed explanations.

Native foreign execution, hours-long service retention and held-out
quality/calibration remain open. `RuntimeReady=false`.
