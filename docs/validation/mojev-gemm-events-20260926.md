# MoJev CUDA event profiling and GEMM tuning

CUDA events now provide usable GPU timing attribution on this host. A measured
GEMM thread-layout change reduces full-request latency by 19–25% while retaining
identical outputs. This follows `a4b59c61`; the earlier Nsight Systems capture
still contains no kernel records.

## Diagnostic event timer

`nvidia.NewLaunchTimer(capacity)` owns `capacity+1` timing-enabled events, bounded
to 512 commands. `Measure` validates every launch before submission, rejects
active graph capture and runs on the default stream under the existing pinned
thread/driver lock. It returns one event interval per command. Failed launches
or event operations drain possible queued work; failed drains disable reuse.
`Close` synchronises before destroying events and retains failed handles for
retry. Owners must close before global runtime shutdown.

This is an explicit diagnostic API; ordinary inference never allocates its
events. Event recording adds overhead and can include GPU idle time waiting for
host submission, especially for small kernels. Intervals are not pure hardware
instruction time or a replacement for uninstrumented end-to-end benchmarks.

`TestMoJevNVIDIAKernelTiming` hash-checks the same approved checkpoint/tokenizer,
executes a supplied request, then restores and replays its final bounded tree
with events. One warmup and five measured replays are used. Each replay must
produce bit-identical hidden output. Multi-question/multi-group inputs report
**only the final group**, not the whole request. The supplied four workloads
produce final-group lengths of 41, 61, 37 and 142 rows, each with 397 launches.

## Measured bottleneck

Before tuning, GEMM contributed 85–92% of summed event intervals. Aggregate
milliseconds for the final group:

| Workload | All events before | GEMM before | All events after | GEMM after |
|---|---:|---:|---:|---:|
| Two choices | 44.67 | 40.96 | 36.37 | 32.62 |
| Eight choices | 46.95 | 41.78 | 38.78 | 33.55 |
| Two questions, final group | 43.77 | 40.39 | 35.20 | 31.81 |
| Longer context | 91.96 | 78.38 | 68.07 | 54.57 |

Three initial experiments passed their kernel comparisons but regressed
whole-request latency and were reverted:

| Experiment | Two choices ms | Eight choices ms | Two questions ms | Longer ms |
|---|---:|---:|---:|---:|
| Initial baseline | 46.19 | 48.06 | 89.54 | 92.96 |
| 16×64 row tile | 59.03 | 76.07 | 114.37 | 148.13 |
| Padded A shared-memory stride | 60.32 | 62.24 | 118.58 | 115.49 |
| 32×128 column tile | 56.34 | 58.09 | 112.53 | 101.72 |

The accepted kernel keeps the 32×64 output tile and 32-wide K panels, but uses
128 threads with sixteen accumulators each, instead of 256 threads with eight.
Each thread owns four rows spaced eight apart and four columns spaced sixteen
apart. Shared-memory loading strides change to 128; bounds handling, K reduction
order, `fmaf` operations and F32 storage remain unchanged. Other kernels retain
256-thread launches. No additional device/host model scratch is reserved.

## Paired uninstrumented benchmark

Same i7-12700, RTX3060, driver580.173.02, Go1.26.3, Linux/amd64 and
`GOMAXPROCS=6`. Existing approved model revision
`0c8695b6252f4205907433d4e196a94f032e60c3`, safetensors SHA-256
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
One model process at a time; one warmup and five samples. Parsing, tokenisation,
inference and response marshalling are timed. Model preparation, hash checking
and HTTP are excluded. These runs do not use the event timer.

| Request | Before ms | After ms | Change |
|---|---:|---:|---:|
| Two choices | 46.08 | 37.47 | -18.7% |
| Eight choices | 49.68 | 39.75 | -20.0% |
| Two questions | 89.09 | 72.20 | -19.0% |
| Longer context | 92.51 | 69.08 | -25.3% |

All saved responses match exactly. An earlier accepted-layout run measured
37.23/39.72/71.93/69.08 ms. No statistical significance claim is made.

## Verification

- Released GPU oracle/race passes with unchanged maximum errors: logits
  `4.7087669e-5` against `3e-4`; hidden `7.8797340e-4` against `2e-3`.
- Exact group/branch comparisons, permutations, cancellation and recovery remain
  enabled. The 512-path/4096-total ordinary three-round admission test passes
  with eight serialized callers; peak RSS 6,094,724 KiB. Existing SIMD full-race
  timeout limitations are unchanged by this GPU-only pass.
- Direct and batched GEMM tests cover 1/7/31/32/33/63/65/512 rows, 17/63/64/65/
  67/127/129 columns and K panel tails. Memcheck reports zero errors; racecheck
  reports zero hazards for direct and tree kernels.
- Event timer mock tests cover ordered launches, context/thread locks, finite
  timings, partial creation, start/mid-record/launch/sync/elapsed failures,
  failed-drain poisoning and retryable close. The locked constructor, validators,
  close, Measure and cleanup helpers have 100% scoped coverage.
  The public constructor's successful path is exercised on hardware.
- Independent focused review found no coverage hole, race or arithmetic-order
  change in the new GEMM mapping, and no event-owner lifecycle defect.
- Whole-tree CPU race exits0; focused races, vet/build, ARM64/RISC-V test-binary
  cross-builds and PTX byte-reproducibility pass. Native qualification is RTX3060
  on amd64; foreign builds do not qualify runtime performance.

Run the diagnostic probe with:

```sh
GOMAXPROCS=6 GO_PHERENCE_MOJEV_TIMING=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  GO_PHERENCE_MOJEV_TIMING_WORKLOADS=/path/to/workloads.json \
  GO_PHERENCE_MOJEV_TIMING_REPORT=/path/to/timing.json \
  go test ./model/mojev -run '^TestMoJevNVIDIAKernelTiming$' -count=1 -v
```

Evidence, rejected experiment scripts, preserved baseline binaries and event
samples are under `/workspace/tmp/mojev-event-timing-20260926`. Wider admission,
independent long-input accuracy, failure coverage and held-out quality remain
open. `RuntimeReady` stays false.
