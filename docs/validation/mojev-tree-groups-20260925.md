# MoJev bounded candidate groups

Both accelerated scorers now split oversized question trees into candidate
groups that fit existing scratch. They no longer discard ancestor sharing for
an entire request because one tree is too large. This follows
[GPU tree execution](mojev-ptx-tree-20260925.md), published as `b7fdf685` with
CI `36201906972` passing.

All candidate paths are validated first. A greedy group contains the common
state/question prefix plus as many whole candidates as fit. The next group
repeats that prefix but never splits a candidate. Candidate order and local
positions remain unchanged. A candidate filling the entire available suffix
runs as a singleton group. Every candidate path must still fit the constructor's
3–512-token capacity; the reference request limit remains 4096 total tokens.

No kernel, precision, tolerance or persistent scratch budget changes in this
pass. Per-question output rows are allocated once, then filled from groups.
Only the current group's rows, ends and masks are visible to execution. The
shared grouping helper allocates nothing.

## Oversized-tree measurements

These are **encoded requests**, excluding tokenisation and JSON, unlike the
four full-library workloads used in earlier reports. State/question lengths are
3/4 tokens. Eight candidates each have 40 tokens (327-token whole tree); 64
candidates each have six tokens (391-token whole tree). Scorers reserve 256
tokens. Before this change those requests used 8/64 separate branches; after,
each uses two groups.

Same approved checkpoint and host: i7-12700, RTX 3060, driver 580.173.02,
Go 1.26.3, Linux/amd64, `GOMAXPROCS=6`. Checkpoint revision
`0c8695b6252f4205907433d4e196a94f032e60c3`; SHA-256
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
One model process at a time, one warmup, five timed samples; preparation and
hash checking are excluded. Baseline is the saved `b7fdf685` executable.

| Backend / candidates | Before ms | Grouped ms | Allocations before → grouped | Bytes before → grouped |
|---|---:|---:|---:|---:|
| PTX / 8 | 368.05 | 198.43 | 283 → 76 | 134,808 → 38,336 |
| PTX / 64 | 2171.95 | 229.20 | 2,050 → 76 | 979,256 → 40,584 |
| SIMD / 8 | 2050.50 | 1811.98 | 205 → 49 | 145,656 → 42,896 |
| SIMD / 64 | 7509.27 | 2188.52 | 1,340 → 44 | 1,041,736 → 42,568 |

Values are medians. Saved logits match exactly. CPU timings remain sensitive
to host scheduling; no significance result is reported. The smaller SIMD gain
for eight long candidates is expected because projection/attention work still
scales with candidate tokens; grouping removes repeated prefix and launch work.
No new peak-RSS or long-term retention result is established here.

## Worker-count experiment

The existing full-request SIMD tree binary was also measured at
`GOMAXPROCS=1/2/3/6`, which changes both available Go parallelism and the worker
cap. One warmup/five samples per request gave these median milliseconds:

| GOMAXPROCS | Two choices | Eight choices | Two questions | Longer context |
|---:|---:|---:|---:|---:|
| 1 | 610.62 | 1025.84 | 1100.83 | 1797.40 |
| 2 | 360.95 | 540.05 | 683.13 | 1162.24 |
| 3 | 301.15 | 432.57 | 556.06 | 928.58 |
| 6 | 235.99 | 363.41 | 485.95 | 786.23 |

Six performed best in this bounded host experiment. Scheduling code is unchanged;
this does not establish the best count on other hosts or under contention.

## Verification

- Released SIMD and PTX tests compare 2/8/64-candidate trees and 63 unequal-length
  candidates spanning multiple groups against independent separate execution.
  Logits are exact, including reverse permutation and singleton groups.
- Original eight pinned F32 cases, sibling/question isolation, four concurrent
  callers, repeatability and output ownership remain enabled. Combined released
  race passes; maximum errors and fixed oracle tolerances are unchanged.
- Group-boundary tests cover exact fits, singleton groups, mixed lengths,
  progress and zero allocations. All IDs/lengths are checked before grouping.
- Whole-tree CPU race suite exits 0; focused races, vet/build and ARM64/RISC-V
  cross-builds pass. No foreign native execution is inferred.
- Focused independent review found no issue with progress, bounds, mask reuse,
  order or stale storage.

Evidence and benchmark sources are in `/workspace/tmp/mojev-workers-20260925`.
GPU kernels are unchanged from the preceding zero-error sanitizer pass. Wider
capacity/concurrency admission, cancellation, coverage targets and held-out
quality/calibration are still open; `RuntimeReady` remains false.
