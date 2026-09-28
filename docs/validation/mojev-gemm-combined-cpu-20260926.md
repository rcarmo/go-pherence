# MoJev combined amd64 GEMM bookkeeping optimization

## Adoption

Adopt the combined four-step reduction unroll and shared A-row offset. Rui
explicitly requested cumulative correctness-qualified optimizations be retained
rather than rejected solely because shared-host timings are inconclusive.

The earlier [separate trials](mojev-gemm-trials-20260926.md) remain preserved,
including their unfavorable observations. This combined implementation was tested
as a new candidate against `0f6f0a7e`; it is not a claim that either earlier timing
result was wrong. The current evidence establishes numerical equivalence and
less integer bookkeeping, **not a statistically established runtime speedup**.

This is CPU work only. The separately preserved combined GPU candidate is not
adopted by this change, and the GPU hold remains active.

## Change and tradeoff

`backends/simd/runtime/gebp_amd64.s` still computes a 6×16 tile with AVX2/FMA.
It now consumes four K values per main-loop iteration instead of two and keeps
six A-row bases fixed, using one shared 64-bit byte offset in broadcasts.

- Over four K values, pointer/offset ADD instructions fall from 14 to 2;
  loop-counter updates and main-loop branches are halved.
- The twelve accumulators retain exactly the original sequential FMA order.
  Existing separate alpha multiplication and output addition remain unchanged.
- A two-step remainder followed by a one-step remainder covers all K residues;
  K0 goes directly to the original store semantics.
- Checked wrappers, input/output validation, weight packing, fallback dispatch,
  projection padding, six-worker cap and ownership are unchanged.
- amd64 kernel text grows from757 to1184 bytes (+427 bytes). Indexed addressing
  and greater instruction-cache footprint are real tradeoffs; fewer integer
  instructions alone do not prove a runtime gain.
- No allocation, retained-workspace or loading-peak reduction is claimed.
  ARM64, RISC-V and generic kernels are unchanged.

A bounded delegated judge review of the explicit instruction/register description
found no new correctness defect. Local source and emitted-instruction inspection
confirmed fixed R8–R13 row bases plus AX indexing, 64-bit updates and 4/2/1 tail
coverage. The delegated review was not an independent checkout audit. Existing
caller contracts reject malformed dimensions before entering assembly.

## Correctness and resource gates

- Native reduction-order/exceptional-value and packed-only API tests under race,
  count10: passed. Cases cover K0–17,31–33,127–129,1024,3584; alpha0/1/−0.75;
  padded/unaligned buffers, guards, source immutability, aliases and allocation
  assertions. Per-output reference uses sequential F32 FMA and separately rounded
  alpha/output operations. NaNs compare by classification, not payload/sign.
- Whole-tree NVIDIA-disabled CPU race: exit0,189 packages, wall5m27.51s.
- Candidate capacity512 grouped released-model race: passed355.13s, Go test total
  356.500s. Independent4096-total-token reference, hidden rows, changed/reordered
  concurrent requests and retained-output ownership passed.
- Errors unchanged: logits `1.4007091522216797e-6`, changed logits
  `4.647299647331238e-7`, hidden `5.7220458984375e-5`. Gates stay `3e-4` / `2e-3`.
- Post-concurrency live heap is62,136 bytes below warm baseline. Race peak RSS
  11,120,472KiB, zero swaps. This is not ordinary memory admission evidence.
- `go vet ./...`, `go build ./...`, docs/layout checks: passed.
- Runtime/Qwen Linux ARM64 and RISC-V test cross-builds: passed. This does not
  qualify native execution there. No new production Go branch was added, so
  Go statement coverage cannot measure the changed assembly; native differential
  tests provide that coverage.

The previously completed SIMD60 run remains evidence for `0131934d`, not a
retroactive60-round run of this candidate. Likewise, JevBench public results
remain pinned to their recorded revision; the benchmark was not rerun or tuned.

## Measurement

Go1.26.3/Linux amd64, reported Intel i7-12700, six available CPUs,
`GOMAXPROCS=6`, NVIDIA disabled. Same approved MoJev checkpoint revision
`0c8695b6252f4205907433d4e196a94f032e60c3`, four asset hashes checked by each
request process. Only one checkpoint process at a time. Baseline and candidate
binaries, patches and host load/pressure observations are preserved.

Six interleaved ABBA samples per variant,200ms per case, include K128/1024/3584
microkernels and five representative packed projection shapes. Geomean time is
4.75% lower, but **none of the eight individual comparisons is statistically
significant**. Some confidence intervals reach59%. The M126 projection median
moves adversely (2.002→2.483ms, p=0.132), while other cases move favorably.
All report0 B/op and0 allocs/op. All observations remain in evidence.

The four existing capacity256 request workloads were run in six ABBA processes
per variant, one warmup and five timed calls per workload. Loading is excluded.
All288 actual warmup/trial responses match exactly, with request hashes checked.

| Workload | Baseline median ms | Combined median ms |
|---|---:|---:|
| Short two choices | 214.7 | 215.6 |
| Short eight choices | 313.2 | 319.1 |
| Two questions | 412.1 | 420.2 |
| Longer two choices | 703.8 | 708.1 |
| Geomean | 373.7 | 378.2 |

Request geomean is1.21% higher; no individual timing change is significant.
This is neither a proven speedup nor a reproducible regression. The change is
retained under Rui's stated cumulative policy with the uncertainty explicit.

Allocated-byte/count geomeans are−0.70%/−0.16%; no allocation benefit is
attributed to assembly bookkeeping. Ordinary peak RSS ranges overlap:
4,988,800–4,991,744KiB before,4,985,984–4,991,872KiB after.

Separate normal-rate profiles show total CPU samples24.97→24.78s and GEMM
samples18.59→18.79s. GEMM remains roughly75% of sampled CPU; this single profile
pair does not establish a speed benefit. `alloc_space`, `alloc_objects`,
`inuse_space` and `inuse_objects` reports are retained. Profiled runs are excluded
from request timing comparisons; heap profile sampling is not a zero-allocation
proof. Loading/packing code is unchanged and received no new cold-cache
qualification. Broader model generations, decode/training and long-running
service workloads were not rebenchmarked; the shared kernel still passes the
whole-tree ordinary CPU tests under race.

## Reproduction and evidence

Evidence: `/workspace/tmp/mojev-gemm-combined-cpu-20260926`. Original assembly,
baseline/candidate executables, generated candidate source, disassembly,
interleaved measurements, exact-response check, host observations, profiles and
all gate logs are preserved. Executables remain local with hashes; the attached
archive contains reports, source/patches, profiles and raw logs.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test -race \
  ./backends/simd/runtime -run 'Test(GebpAMD64|SgemmNTPackedOnly)' -count=10
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test \
  ./backends/simd/runtime -run '^$' \
  -bench 'Benchmark(GebpAMD64|PackedGEMMProjection)$' \
  -benchtime=200ms -benchmem -count=6
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test -race \
  -p=2 ./... -count=1 -timeout=180s
```

No new GPU execution, frozen evaluation or service restart occurred.
`RuntimeReady=false`; native foreign execution, structured-state/long-context
coverage, broader quality and hours-long retention remain open.
