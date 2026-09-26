# MoJev combined GPU attention and recurrence

Warp-local attention and precomputed recurrence parameters improve different
parts of the GPU scorer. Combined, they reduce three measured workload medians
by 1.16–2.75%, with exact responses and unchanged request bytes. This candidate
is preserved on `experiments/mojev-gpu-combined`; main dispatch remains unchanged
while a maximum-capacity CUDA racecheck device failure is unresolved.

## Changes and numerical order

`mj_tree_attention_warp` assigns one query/head to each warp (eight warps per
256-thread block). It reproduces the existing eight 32-element partial
reductions, then reduces those partials in the same tree. Online softmax, gating
and ancestor/candidate visibility are unchanged. It removes block-wide barriers
from the key loop.

`mj_delta_params` computes each token/head's sigmoid and decay once, in the
existing alpha/beta scratch. `mj_tree_delta_prepared` consumes those values
instead of recomputing them for 128 value rows. Projection overwrites this
scratch before each layer/replay; launch order is projection → preparation →
recurrence. Cancellation still drains submitted work. The original kernels
remain available for comparison and compact-branch execution. The extra prepare
launch raises a representative 397-command replay to415 commands, within the
existing512-command storage.

Both changes retain exact F32 operations and compile with `--fmad=false`; no
approximation or tolerance widening is used. Focused read-only review found no
shuffle divergence, barrier, scratch-ordering or ownership bug in these changes.

## Cumulative experiment

Baseline `661a3255`, Go1.26.3/Linux amd64, i7-12700, `GOMAXPROCS=6`, RTX3060,
driver580.173.02. One approved checkpoint/backend process at a time. Checkpoint
weights SHA-256:
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
Probe asset hashes are checked before loading.

The same PTX module includes reference and new kernels for all four binaries.
Only dispatch differs: baseline, attention-only, recurrence-only and combined.
Six interleaved processes per variant, one warm-up and five timed calls per
workload. All96 workload example comparisons are exact; each process separately
checks exact response repeatability. No other checkpoint process runs alongside
these measurements.

| Workload | Baseline ms | Attention ms | Recurrence ms | Combined ms |
|---|---:|---:|---:|---:|
| Short, two choices | 37.55 | 37.36 | 37.22 | 37.11 |
| Short, eight choices | 40.03 | 39.81 | 39.66 | 39.57 |
| Two questions, two choices | 72.48 | 72.48 | 72.53 | 72.29 |
| Longer, two choices | 69.90 | 68.74 | 69.26 | 67.97 |

Combined versus baseline: −1.16% (p=.026), −1.16% (p=.004), inconclusive
(p=.180), −2.75% (p=.002), respectively; n=6 process medians. Time geomean is
1.34% lower. Request B/op is unchanged. These gains are small and workload-
dependent, but accumulate: the combined result exceeds either isolated result
on the longer workload.

A one-second collector records CPU/load pressure, cgroup quota and throttling,
and GPU utilisation, temperature, clocks and power during all24 processes.
Recorded throttling deltas are zero; CPU pressure `some avg10` ranges .22–26.62
and GPU temperature47–65°C. Shared-server contention was present; no services or
governors were changed. All samples are retained, including busy intervals.
Collector overhead is common to all variants.

CUDA-event replay gives local attribution (five measured samples after warm-up).
For the longer workload's final142-row group, six attention launches total
4.0944ms before and2.9700ms combined; eighteen recurrence launches total6.7767ms
before and5.9641ms combined, plus .0849ms for parameter preparation. Events measure
only final-group replay, not complete multi-group requests. This attribution is
separate from the full-request timing table.

## Completed gates

- Exact comparison to compact baseline kernels and repeatability: initial
  10-row sibling fixture10 times, then geometries3/10/33/512 three times.
  These cover minimum shape, uneven siblings, parameter-block tail and capacity.
- Released ordinary short/isolation, independent512-path and4096-total grouped
  gates pass; unchanged `3e-4` logits and `2e-3` hidden tolerances.
  Grouped maximum errors: logits `1.34110e-6`, hidden `6.86646e-5`.
  Maximum long hidden error `1.33514e-4`.
- Released race suite covering short, long, grouped concurrency, host lifetime
  and expanded kernel tests passes in292.804s.
- CUDA memcheck passes expanded geometries with zero errors.
- CUDA racecheck passes the initial10-row fixture, three repeats, zero hazards.
- Whole-tree NVIDIA-disabled race exits0; affected package races, vet/build,
  ARM64/RISC-V builds and byte-identical PTX regeneration pass. No native foreign
  execution was performed. A later test-only change passes the actual subtest
  `*testing.T` into the launch helper, fixing misleading parent-FailNow reporting
  on driver errors; it was checked model-free after GPU access was lost.

## Unresolved device failure

The first expanded racecheck invocation was interrupted by the enclosing shell
timeout after the released-race and memcheck stages had already passed. Its log
is retained separately. A standalone rerun failed within the capacity512
`warp_attention` subtest with `cuMemcpyDtoH: error 719` after about60.7 seconds
in that subtest (263.8 seconds for the test). **The reported source line192 is
the compact reference call `mj_attention`, not the new warp kernel launch.**
That subtest first runs the new kernel, then launches multiple compact reference
branches. The original helper used its parent's `*testing.T`, adding a misleading
FailNow warning. The kernel responsible for the device failure cannot be inferred
from the subtest name or the later asynchronous copy error. Racecheck printed
zero hazards but the target failed; this is **not a sanitizer pass**.

Guest kernel journal at `2026-09-26T11:21:52+01:00` records Xid79, "GPU has
fallen off the bus," followed by Xid154, "Node Reboot Required." Then
`nvidia-smi` returned `Unknown Error` and `No devices were found`. This matches
previously recorded evaluator failures that predate the new kernels. It does not
exonerate application code as a trigger. CUDA719 is launch failure; the distinct
CUDA timeout code is702. Instrumentation timeout is therefore a hypothesis,
not a diagnosis. No incident-time temperature/power recording was captured;
47–65°C readings came from the earlier benchmark matrix.

A read-only audit found no concrete indexing, shuffle-mask, barrier or lifetime
bug in the candidate under its validated launch geometry. Current copies and
launches retain their OS-thread pin/context lock; the prior context fix is in
ancestry and model-free regressions pass. Offline sm86 compilation shows48
registers, no spills and no shared-memory barriers for warp attention. These
checks cannot exclude an instrumentation/driver, PCIe/passthrough, power or
application-triggered device failure.

Test diagnostics now log the actual kernel, tree/reference mode, dimensions
and launch geometry and synchronise each tested launch explicitly. These are
model-free-checked diagnostic changes; no GPU rerun followed the fault. A bounded
NVIDIA safe-mode diagnostic collection timed out and left a partial readable
archive. No GPU reset, VM reboot or unrelated service change was attempted.

Resume only after GPU access is restored through authorised infrastructure
recovery. Recheck baseline and combined ordinary kernels, then investigate the
512-row instrumented case with a bounded run. Do not silently skip that shape,
widen tolerance or mark zero displayed hazards as a pass. The tested source,
all individual/combined benchmarks and failure logs are retained; this hold is
for unresolved device behaviour, not a rejection of small cumulative gains.

Evidence: `/workspace/tmp/mojev-gpu-combined-20260926`, including `matrix.ts`,
`variant.ts`, raw host/response JSON, `benchstat.txt`, CUDA-event reports, tests,
sanitizer logs and `device-failure-host.txt`. `RuntimeReady=false`.
