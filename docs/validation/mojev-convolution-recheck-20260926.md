# MoJev convolution impact recheck

## Finding

**Keep single-pass SIMD convolution.** The isolated recheck confirms substantial
local savings and lower convolution CPU cost inside real requests. Whole-request
timings cannot reliably resolve its small contribution; that is not evidence
that the optimization has no effect.

This recheck responds to Rui's request after native-tile padding was adopted.
All three variants use baseline `0131934d`, including the same padding,
tokenizer, recurrence, packed weights and six-worker scheduling. No production
code changed during this comparison. Preserved GEMM loop candidates were not
applied, and GPU execution remains on hold.

## Controlled variants

Go overlays substitute only convolution implementation plus a common signature
and call site. All variants receive the same idle `padIn[:6144]` scratch argument
so the two-pass variant is not uniquely charged for an extra argument. Scalar
and single-pass do not use that scratch. It is projection-owned, already
allocated, and no projection jobs are pending when convolution runs.

1. Scalar: original channel loop, explicitly rounded F32 multiply then add.
2. Two-pass SIMD: the `241cb34e` `VecMul` then `VecAdd` implementation.
3. Single-pass SIMD: current `VecMulAddTo`, with the common unused scratch
   parameter added only in the overlay.

Tap order and ancestry are identical. Each overlaid test binary passes the
existing scalar-reference bitwise convolution test count10, including source
preservation, sentinels, sibling independence and zero allocations. That test
runs native and disabled dispatch. Final JSON equality supplements rather than
replaces this internal check. Same-source builds can still differ in code layout
and inlining; these results are not a byte-identical production-binary comparison.

## Measurements

Go1.26.3/Linux amd64, reported i7-12700, six available CPUs, `GOMAXPROCS=6`,
NVIDIA disabled. Approved revision
`0c8695b6252f4205907433d4e196a94f032e60c3`, all four asset hashes checked per
process. One checkpoint process at a time. No affinity/governor change, other
service restart, GPU execution or frozen evaluation.

### Convolution row

Ten interleaved samples per variant, 300ms each, ordinary magnitudes (subnormal
correctness remains in tests), 6,144 channels and four taps:

| Implementation | Median | Versus scalar | Allocations |
|---|---:|---:|---:|
| Scalar | 17.147 µs ±7% | — | 0 B / 0 |
| Two-pass SIMD | 3.823 µs ±2% | −77.70%, p<0.001 | 0 B / 0 |
| Single-pass SIMD | 2.151 µs ±2% | −87.46%, p<0.001 | 0 B / 0 |

Single-pass is **43.75% faster than two-pass** (p<0.001). This confirms the local
benefit; the percentage differs from earlier runs because these are fresh,
interleaved measurements rather than pooled historical samples.

### Full requests

Twelve independent processes per variant. Six permutations of the three
variants, followed by the permutation list in reverse order, distribute process
positions and preceding variants. Each process loads once, warms each of four
existing workloads once and times five calls. Each process's workload median is
one statistical sample; the five calls are not treated as independent processes.
One-second host observations are preserved. Profiles are separate from timing.

These are text-scoring requests with 41, 61, 61 and142 input tokens and **zero
output tokens**, not generative prefill/decode throughput. Loading is excluded.

| Request | Scalar ms | Two-pass ms | Single-pass ms |
|---|---:|---:|---:|
| Short, two choices | 225.35 | 218.31 | 233.61 |
| Short, eight choices | 346.92 | 321.26 | 321.76 |
| Two questions | 451.87 | 449.13 | 439.07 |
| Longer, two choices | 766.92 | 722.05 | 745.85 |
| Four-workload geomean | 405.7 | 388.3 | 396.1 |

All **864 actual warmup/trial responses are exact**, with request hashes and
names checked. Process-median ranges are wide; for example longer-request
ranges are scalar716–937ms, two-pass691–963ms and single-pass672–941ms. Full
ranges and benchstat confidence intervals are in the evidence.

- Scalar → two-pass geomean: −4.28%. Only the longer workload comparison is
  significant (−5.85%, p=0.033); this is one comparison in an exploratory set.
- Scalar → single-pass: −2.37%; no individual workload difference significant.
- Two-pass → single-pass: +2.00%; no individual workload difference significant.

The last number is not evidence of a reproducible single-pass regression and
must not override the measured local gain. Equally, these geomeans do not prove
a specific full-model speedup. The local optimization remains adopted.

Allocated bytes/counts show no significant differences in this matrix. Process
peak RSS ranges overlap at roughly4,985,728–4,992,000KiB; no RSS gain is claimed.
The product scratch was already borrowed, so avoiding its traffic does not save
an allocated weight/workspace buffer.

### CPU profiles inside real requests

Three separately profiled processes per variant, rotated order, same workloads
and call counts. All216 profiled responses also match the unprofiled reference.

| Variant | Total sampled CPU | Convolution cumulative CPU | Convolution share |
|---|---:|---:|---:|
| Scalar | 77.24 s | 2.11 s | 2.73% |
| Two-pass SIMD | 75.95 s | 0.57 s | 0.75% |
| Single-pass SIMD | 74.94 s | 0.40 s | 0.53% |

Scalar convolution samples per run are0.79/0.57/0.75s; two-pass0.11/0.22/0.24s;
single-pass0.11/0.16/0.13s. Summed convolution samples drop **81.0% versus
scalar** and **29.8% versus two-pass**. Sampling makes these attribution estimates,
not precise kernel timers or independent end-to-end benchmarks.

Convolution starts at only2.73% of sampled CPU. Saving most of that cost explains
why the local improvement is real while full-request timing remains difficult
to separate from host noise. CPU-time share is not wall-time share: parallel GEMM
and serial orchestration prevent an exact Amdahl wall-time prediction.

Heap profiles (`alloc_space`, `alloc_objects`, `inuse_space`, `inuse_objects`)
are preserved. No loading, allocation or retained-memory improvement is inferred
from these convolution-only comparisons.

## Scope and evidence

A delegated judge review of the experimental design recommended identical
signatures, more independent repetitions and internal bitwise checks. All three
were applied. This was a design review, not a separate implementation audit.
The recheck is amd64 CPU performance evidence; native ARM64/RVV remains open.
Production numerical/race qualification remains in the previous convolution and
padding reports. No new production code or numeric tolerance changed here.

Evidence: `/workspace/tmp/mojev-conv-recheck-20260926`, with exact overlay sources,
JSON mappings, preserved binaries, workload observations, host samples, all raw
benchmark distributions, profile files, and response-verification scripts.
Binaries are hashed and kept locally; the downloadable archive omits executables.
This report does not close held-out quality, hours-long retention or GPU gates.
