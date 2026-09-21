## CPU allocation and SIMD profiling

Follow-on: the legacy `gpu/` CPU attention helper now reuses checked SIMD GQA with one score row per call (128 allocations/32,768 bytes to one/256 in a synthetic 32-query workload). Scalar-oracle/repeated-call tests also caught stale output accumulation. See [the follow-on audit](../validation/repository-safety-followon-20260919.md); this is CPU helper work, not GPU execution.

Top-p sampling allocated six MiB per 128K-vocabulary draw, mostly to sort a
permutation and copy candidate/weight arrays. Top-k allocated 46 objects for 40
survivors. Those are avoidable costs before any new assembly is needed.

The source scan covers every tracked Go file across build tags. CPU profiles
cover synthetic sampling, tensor fusion, FFT/mel and tiny-scorer workloads on the
Intel Core i7-12700 host with Go 1.26.3. No model weights, final-test rows, GPU
execution or service restart were used. This is repository-wide lexical coverage
with sampled dynamic coverage, not a profile of every model or platform.

## Measured sampling change

The top-k heap now uses typed sift operations instead of boxing candidates through
`container/heap`, and its capacity is capped at vocabulary length even when the
caller supplies an enormous K. Softmax reuses its scaled-logit array for weights.
Top-p sorts indices with `slices.SortFunc` and applies permutation cycles in place
instead of allocating two output arrays. Input logits remain untouched.

Weights and their total are computed in the original order. Sorting before
softmax or substituting approximate SIMD exponentials would alter rounding and
possibly threshold decisions, so neither was done. Differential tests compare
complete results against an independent full-sort/separate-array oracle over
ties, NaN/-Inf exclusions, +Inf winners, lengths 1--513, several temperatures,
K/P combinations and draws at/beyond the endpoints. An allocation regression
bounds top-k/top-p to at most three objects per call.

Three interleaved runs per binary used CPU 2, `GOMAXPROCS=1` and 300ms benchmark
windows. The baseline was compiled with a Go source overlay from the committed
pre-change sampling file; both binaries used the same benchmark definitions.

| Benchmark | Before median | After median | Bytes before / after | Allocs before / after |
|---|---:|---:|---:|---:|
| Top-k 32K, K=40 | 86.133 us | 83.805 us | 1,360 / 640 | 46 / 2 |
| Top-p 32K, P=0.9 | 4.571 ms | 4.410 ms | 1,572,920 / 786,432 | 8 / 3 |
| Top-k 128K, K=40 | 276.031 us | 292.114 us | 1,360 / 640 | 46 / 2 |
| Top-p 128K, P=0.9 | 23.329 ms | 22.913 ms | 6,291,512 / 3,145,728 | 8 / 3 |

The allocation reduction is clear: roughly 53% fewer bytes for top-k and 50% for
top-p. Timing is noisy on this shared host, and 128K top-k's median is worse.
There is no general throughput claim. An initial paired-slice `sort.Interface`
version used fewer bytes still but was slower; it was discarded rather than
reported as a speedup. Raw samples retain both exploratory and final runs.

## What the profiles point to next

| Path | Measured result | Next candidate and required checks |
|---|---|---|
| `runtime/sampling` | Before: top-p sorting 49.17% of sampled allocation bytes, softmax 32.88%, full scan 15.58% in the combined top-k/top-p profile | Retained allocation changes above. Exact float64 sampling is not interchangeable with approximate float32 SIMD softmax. |
| `tensor/fuse.go` | Fused interpreter 95.33% flat CPU; `pooledAlloc` 94.99% allocation bytes on AddMul/FusedChain5 | Pattern-dispatch common chains to checked SIMD kernels; caller-owned output/workspace could remove allocation, but retained tensor outputs must never alias recycled scratch. Test broadcasts, DAG reuse, tails and output ownership first. The function named `pooledAlloc` currently allocates; no pool was added. |
| `backends/simd/fft/fft_simd.go` | `forwardRealOpt` 49.11% flat CPU, 93.74% allocation bytes in FFT/power/mel profile | A caller-owned real/imag/output workspace and precomputed bit-reversal/twiddles; preserve float64 operation order or set measured tolerances before SIMD butterflies. The `SIMD` entry-point name does not prove that butterflies use assembly. |
| `model/jevlike` tiny scoring | 1.073ms, 1,143,992 B/op, 4,287 allocs/op; `linearNoBias` 43.08%, TinyScorer.Forward 24.65%, layerNormVector 21.39% of sampled allocation bytes | Per-request vector/normalisation workspaces; do not share mutable scratch across callers. Existing SIMD projection is not the same as allocation-free scoring. |
| `backends/simd/runtime` softmax | At lengths 128/218, scalar about 790/1,268ns, existing SIMD about 274/460ns; both zero allocations | Approximately 2.8x kernel speed ratio on this host. Approximate AVX2/FMA exp has a bounded-error contract, not bitwise parity. Adoption in attention requires caller-level parity, exceptional-input and mask/tail tests. |

The lexical scan covers **2,320 tracked Go files in 161 directories**, including
tests and all architecture tags. It finds 4,342 allocation-candidate lines, 423
scalar-math lines and 351 reduction-looking lines. There are **129 directories
without benchmark declarations** and 66 without ordinary Test declarations; the
latter differs from the host's 71 no-test packages because build tags are not
filtered and non-Test declarations are not execution coverage.

| Source area | Directories | Allocation-candidate lines | Scalar-math lines | No benchmark declarations |
|---|---:|---:|---:|---:|
| Backends | 41 | 438 | 84 | 28 |
| Models | 25 | 2,831 | 286 | 16 |
| Loaders | 11 | 301 | 17 | 6 |
| Runtime | 12 | 234 | 6 | 9 |
| Tensor | 1 | 43 | 14 | 0 |
| Commands | 64 | 493 | 12 | 64 |
| Internal/GPU/half | 6 | 2 | 4 | 5 |

The docs package is the remaining directory. The source inventory does not rank
CLI startup allocations as inference hotspots simply because there are many.

The lexical scan also ranks model families, loaders and CLI/runtime packages by
`make`/`new`/`append`, scalar `math` calls, reduction-looking loops and existing
SIMD calls. Counts are candidates, not heap allocations or hotness. It includes
foreign architecture files without executing them; it cannot infer import aliases,
escape behaviour, aliasing or FMA/reduction-order suitability. Packages without
benchmark declarations remain explicitly visible in its output.

## Tests and reproduction

```sh
mkdir -p .gotmp /tmp/go-pherence-profiles
bun scripts/profile-source.ts > /tmp/go-pherence-profiles/source-inventory.json
bun test scripts/profile-source.test.ts
GOTMPDIR=$PWD/.gotmp GO_PHERENCE_DISABLE_NVIDIA=1 go test ./runtime/sampling \
  -run '^$' -bench 'Benchmark(TopK|TopP)128K$' -benchmem -benchtime=2s \
  -cpuprofile=/tmp/go-pherence-profiles/sampling.cpu \
  -memprofile=/tmp/go-pherence-profiles/sampling.mem \
  -o /tmp/go-pherence-profiles/sampling.test
go tool pprof -top /tmp/go-pherence-profiles/sampling.cpu
go tool pprof -top -alloc_space /tmp/go-pherence-profiles/sampling.mem
GOTMPDIR=$PWD/.gotmp GO_PHERENCE_DISABLE_NVIDIA=1 go test -race \
  ./runtime/sampling ./tensor ./backends/simd/runtime ./backends/simd/fft
GOTMPDIR=$PWD/.gotmp GODEBUG=cpu.all=off go test \
  ./runtime/sampling ./backends/simd/runtime
```

A missing portable test branch was exposed: the blocked-FMA parity test expected
assembly output even when the ISA gate intentionally made the low-level wrapper
a no-op. It now asserts unchanged destination on unavailable ISA and retains the
numeric parity/tail assertions when assembly is available. This does not add a
scalar implementation to a hardware-only wrapper or skip its unavailable branch.
Both native and CPU-features-disabled targeted runs pass. The final NVIDIA-disabled
whole-tree race sweep also passes: 90 packages with tests and 71 with no tests.
Vet, host build and docs/link checks pass; Bun passes 22 tests across nine files.
ARM64 and RISC-V whole-tree builds pass as compilation only. The lexical profiler has tests
for comments/literals, line numbers, generated files and test/benchmark gaps.

Raw profiles, binaries, source inventory, failed exploratory checks and benchmark
samples are retained in `/workspace/tmp/go-pherence-perf-20260919/`. The frozen
Qwen evaluation executable and its 676 outcomes were not used or replaced.
The [repository safety audit][audit] remains separate: passing a microbenchmark
or identifying a SIMD opportunity does not close its source/hardware gaps.

[audit]: ../validation/repository-safety-audit-20260919.md
