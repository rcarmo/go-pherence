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
| Owned weight transfer during loading | `661a3255` | About 3.01 GB less CPU-load allocation; measured SIMD peak RSS 5.78–5.90 → 4.76 GiB. [Ownership and load report](mojev-owned-load-20260926.md). |
| Four-tap SIMD convolution | `afb61da5` | Exact separate multiply/add, no new scratch allocation; microbenchmark −80.73%. CPU grouped race interrupted by maintenance then passed on `298161d2`; [qualification](mojev-conv-simd-maintenance-20260926.md). Full-request improvement is inconclusive; retained for cumulative benefit. |
| Single-pass convolution multiply/add | See [qualification](mojev-muladd-20260926.md) | Removes product-row traffic while preserving both F32 roundings; another −25.46% on the convolution microbenchmark, zero allocations. Candidate grouped race passed363.16s; all576 timed/warmup responses exact. Two request matrices disagree (+16.39% / −3.18%); no whole-model speedup or retained-memory claim. |

## JevBench v1.4.0 public comparison

The [fresh MoJev / historical GSO comparison](mojev-jevbench140-vs-gso-20260926.md)
reports103/231 correct for the current native CPU stack versus audited GSO196/231.
MoJev refuses55 over-capacity branches and35 structured states; on the common141
valid inputs it scores103 versus GSO136. This is a public, now-observed benchmark,
not an official sealed score or intrinsic model-only comparison. No GPU was run;
a fresh GSO GPU rerun awaits explicit hold clearance. Existing output-preserving
optimizations remain adopted and `RuntimeReady=false` remains unchanged.

## Bounded SIMD60 reuse completed

The [retention follow-up](mojev-retention-followup-20260926.md) now includes a
successful60-round ordinary CPU run on `0131934d`:2220.96s, retained outputs,
eight callers and cancellation/recovery passed. Maximum live-heap increase over
warm baseline174,288 bytes, two goroutines, zero swaps. This closes the prior
interrupted SIMD60 gate, not hours-long operation or GPU qualification.

## Convolution rechecked on the cumulative build

The [isolated recheck](mojev-convolution-recheck-20260926.md) holds all other
`0131934d` optimizations constant. Ten samples measure scalar17.147µs,
two-pass3.823µs and single-pass2.151µs: single-pass is87.46% faster than scalar
and43.75% faster than two-pass, still zero-allocation. Three real-request CPU
profiles per variant attribute2.11/0.57/0.40s to convolution. All864 timed/warmup
responses and216 profiled responses are exact. Twelve request processes per
variant leave single-pass full-request differences statistically unresolved;
**the demonstrated convolution optimization remains adopted**.

## Architecture-specific projection padding

The [row-padding qualification](mojev-row-padding-20260926.md) replaces the
cross-platform 12-row multiple with the actual GEMM tile (6 on amd64, 4 on
other targets). It removes dummy-row work and sometimes padding copies without
changing real-token arithmetic. Old-12 bitwise comparisons and released grouped
race pass; 576 request responses are exact. Two request matrices have latency
geomeans −5.65% / −4.54%, with differing individual significance and no claimed
whole-request allocation or RSS improvement. Native foreign execution remains
open. The original GEMM assembly and six-worker cap are unchanged.

## Packed GEMM experiments not adopted

[Two amd64 loop trials](mojev-gemm-trials-20260926.md) preserve exact arithmetic
but do not establish a reproducible improvement. Four-step unrolling has mixed
results; a shared A offset's initial gains disappear in interleaved measurements.
Both patches and binaries remain in evidence. Production assembly is unchanged;
only stronger native reduction-order/edge tests and projection benchmarks are
retained. This is not a reversal of any qualified local optimization.

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
| Warp-local GPU attention + precomputed recurrence | Tested separately and combined across24 load-monitored processes. Combined medians −1.16%/−1.16%/inconclusive/−2.75%; exact responses, independent long/grouped references and released races pass. | Preserved in Git as `a3fffac4`, branch `experiments/mojev-gpu-combined`. Expanded memcheck and small racecheck pass, but512-row racecheck failed with CUDA719/Xid79. Failure surfaced at compact reference `mj_attention` download inside the warp test, not conclusively in the new kernel. Diagnostic follow-up `a8b52e4b`; [bus-loss audit](mojev-gpu-bus-loss-audit-20260926.md). Main dispatch held for unresolved device failure, not for lack of gain. [Corrected branch report](https://github.com/rcarmo/go-pherence/blob/a8b52e4b/docs/validation/mojev-gpu-combined-20260926.md). |
| CPU padding/stride/prefetch/unroll/K-block variants | Mixed/no consistent request gains in prior tests | Preserve assembly and logs; revisit only as specified combinations or changed workloads. `/workspace/tmp/mojev-cpu-tiles-20260926`; prior [record](mojev-long-f32-normalisation-20260926.md). |
| NVIDIA 16×64 and 32×128 GEMM tiles | Slower in the recorded experiment | Different geometry/device may change the result; current 32×64 remains selected. [GEMM record](mojev-gemm-events-20260926.md). |

Workspace artifact paths above are evidence locations, not portable source
checkouts. Four-row recurrence and the combined GPU source are now in Git; older workspace-only prototypes
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
