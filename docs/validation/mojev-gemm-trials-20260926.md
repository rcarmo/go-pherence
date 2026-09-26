# MoJev packed GEMM loop trials

## Result

Neither trial is adopted. Both preserve reduction order, but neither establishes
a reproducible speed improvement. The production amd64 GEBP kernel is restored
byte-for-byte to baseline `1b333eb8`. Only regression tests and benchmarks are
added to the repository. This does not undo the qualified convolution or earlier
memory optimizations.

The latest [CPU profiles](mojev-muladd-20260926.md) attribute about75% of sampled
CPU time to the6×16 AVX2/FMA kernel. Its two-step reduction loop advances six A
row pointers plus the packed-weight pointer on each iteration. Two bounded
experiments tested that bookkeeping without changing packing, scheduling,
accumulator order, FMA use or alpha/output arithmetic:

1. **Four-step unroll:** perform four reductions before advancing pointers,
   retaining the two-step and single-step tails.
2. **Shared A offset:** keep six row bases fixed, use one64-bit byte offset
   in broadcasts, and update that offset instead of six pointers. Reduction
   unroll remains two.

The separate-multiply/add requirement from convolution does not apply to GEMM:
this GEMM kernel already uses FMA, and the trials retain that existing contract.

## Measurements

Baseline `1b333eb8`, Go1.26.3/Linux amd64, reported Intel i7-12700, six available
CPUs, `GOMAXPROCS=6`. NVIDIA was disabled. No checkpoint, GPU, model service or
frozen evaluation was executed. Baseline request binaries were prepared but not
run: the model-free acceptance gate did not justify released-model experiments.

Microbenchmarks cover reduction lengths128/1024/3584. Full packed-GEMM
benchmarks cover `(M,N,K)` of `(42,1024,1024)`, `(42,3584,1024)`,
`(42,1024,3584)`, `(126,1024,1024)` and `(510,1024,1024)`. Packing/setup is
outside timing; full-GEMM benchmarks include checked dispatch. Every benchmark
reports zero bytes and zero allocations per operation.

Six200ms samples per case in each initial series:

| Trial | Result versus baseline |
|---|---|
| Four-step unroll | Time geomean−0.11%; M126 case+6.63% (p=0.002), M510−2.31% (p=0.009); other cases not significant |
| Shared offset | Initial geomean−3.32%; two wider M42 projections−9.22% and−11.21% (p=0.002 each); remaining cases not significant |

The shared-offset candidate then underwent six samples per version in ABBA
process order, with timestamps, load averages and CPU pressure saved before
each process. **None of the eight cases showed a statistically significant
change.** Geomean was+1.24%; some individual intervals were wide (up to±49%).
This does not prove equivalence or establish a regression. It is insufficient
evidence to alter a shared microkernel. Initial favorable results and subsequent
inconclusive results are both preserved.

## Correctness and retained tests

`backends/simd/runtime/gebp_amd64_test.go` adds:

- A native bitwise oracle for sequential F32 FMA, followed by separately rounded
  alpha multiply and output add, across all96 outputs, K0–17 and longer/tail
  lengths through3584, alpha0/1/−0.75, padded strides and unaligned storage.
- Source immutability, full output/padding guards and initialized C values.
- Signed-zero, subnormal, maximum-finite, infinity and NaN checks. NaN payloads
  and signs are not a portable guarantee.
- Allocation-reporting microkernel and representative packed-projection
  benchmarks. The initial evidence calls the latter `BenchmarkMoJevPackedGEMM`;
  its committed generic name is `BenchmarkPackedGEMMProjection`.

Both candidates passed the initial K1–17/longer exact-order oracle count10.
K0 and exceptional-value additions were subsequently tested on the restored
production kernel; they are not retroactive candidate qualification. Final
focused tests include existing packed-only API/error/alias/full-matrix tests
under race count10. Full runtime race, vet and build passed. No production code
changed, so no new released-model, whole-tree race or foreign-native result is
claimed; the previous slice's gates remain the latest production qualification.

A bounded independent judge review of the described address rewrite found no
obvious functional flaw and called out AX liveness,64-bit updates, K0 and IEEE
coverage. It was not a full source audit. The K0/IEEE gaps are covered by the
retained tests. This review does not override the missing performance evidence.

## Evidence and next work

`/workspace/tmp/mojev-gemm-unroll-20260926` preserves the original assembly,
both candidate sources/patches, test binaries, sample logs, host observations,
benchstat reports and review summary. No experiment was discarded by resetting
or rebasing the repository. The downloadable bundle omits binaries, which remain
local with hashes.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test -race \
  ./backends/simd/runtime -run 'Test(GebpAMD64|SgemmNTPackedOnly)' -count=10
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test \
  ./backends/simd/runtime -run '^$' \
  -bench 'Benchmark(GebpAMD64|PackedGEMMProjection)$' \
  -benchtime=200ms -benchmem -count=6
```

Further work should measure cache/panel reuse or projection scheduling rather
than assume loop bookkeeping is limiting throughput. Any packing or blocking
change must preserve reduction order and account for setup, retained weights,
caller ownership, cancellation and actual full-request costs. GPU clearance,
native foreign execution, held-out quality and hours-long retention remain open;
`RuntimeReady=false`.
