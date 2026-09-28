# MoJev single-pass convolution multiply/add

## Scope and decision

Keep the CPU convolution optimization: replace a temporary product write/read
with a checked, single-pass `VecMulAddTo`. Multiplication and addition still
round separately to F32. This is not fused arithmetic, and no tolerance changed.

The 6,144-channel four-tap microbenchmark improves by 25.46%, with zero
allocations. Full-request results disagree across two busy-host matrices;
**end-to-end performance remains inconclusive**. This is a local optimization
retained under Rui's cumulative-optimization policy, not a model speedup claim.

Baseline is `241cb34e` (CI run `36241831361` passed). Candidate changes are the
new SIMD primitive and its Qwen branch-convolution integration. Baseline binaries
were preserved before editing; the request baseline uses a Go overlay containing
both original Qwen production files. No checkout, reset or rebase was used.

## Implementation and numerical contract

`backends/simd/runtime/vec_mul_add.go` accepts equal, non-empty slices. It rejects
partial output/input overlap before mutation, permits exact in-place aliases,
and permits overlap between read-only inputs. It uses no scratch or allocation.

- amd64: eight F32 lanes, separate `VMULPS` and `VADDPS`.
- ARM64: eight F32 lanes, separate NEON `FMUL` and `FADD`.
- Portable fallback and scalar tails: explicit F32 product conversion prevents
  contraction into FMA. The existing `HasVecAsm` flag controls native dispatch.

The convolution keeps ancestry and tap accumulation order unchanged. It no
longer borrows projection padding for products. For a row with all four taps,
this removes 192 KiB of logical temporary-product stores/loads. That is not a
measured DRAM-bandwidth saving. Projection padding remains needed elsewhere,
so there is **no retained-heap or loading-peak reduction claim**.

Tests cover lengths 1–65 and 6,144, all eight pointer offsets, every scalar tail,
output sentinels, exact aliases, overlapping read-only inputs, rejected partial
aliases in both directions, malformed lengths, source preservation and eight
concurrent calls. Deterministic random bit patterns include subnormals, signed
zeros, infinities and NaNs. An independent float64-operation/F32-rounding oracle
and explicit FMA counterexamples verify the separate-rounding contract. Finite
values and zeros compare bitwise; NaN classification is checked without claiming
portable NaN payload/sign selection.

## Measurements

Go1.26.3/Linux amd64, host CPU reported as Intel i7-12700, six available logical
CPUs, `GOMAXPROCS=6`, NVIDIA disabled. Same approved MoJev revision
`0c8695b6252f4205907433d4e196a94f032e60c3`, same four asset hashes, one checkpoint
process at a time. No other model, GPU, frozen evaluation or service was started.

Ten microbenchmark samples per binary, normal profiling disabled:

| Four-tap row | Baseline | Candidate | Result |
|---|---:|---:|---|
| SIMD | 3.817 µs | 2.845 µs | −25.46%, p<0.001 |
| Unchanged scalar control | 17.12 µs | 18.24 µs | +6.52%, p=0.005; host variation |
| Both paths | 0 B / 0 allocs | 0 B / 0 allocs | Exact allocation gate passed |

The request probe uses four existing workloads, capacity256, one warmup and five
timed calls per workload. Each matrix has six processes per version, alternating
ABBA order. Each process checks artifact hashes; loading stays outside request
timing. One-second load, CPU pressure, cgroup and process observations are saved.

| Request-median comparison | Default placement | Explicit `taskset -c 0-5` |
|---|---:|---:|
| Four-workload latency geomean | +16.39% | −3.18% |
| Eight-choice request | +20.49%, p=0.026 | −4.87%, p=0.002 |
| Other three workloads | Not significant | Not significant |
| Allocated-byte geomean | +0.54% | −0.60% |
| Allocation-count geomean | +0.07% | −0.09% |

The affinity mask is the same six CPUs already available to the process; it does
not isolate the workload or remove contention. Initial-matrix load averages
ranged 2.69–5.32, with substantial variability inside each workload. The two
matrices do not establish either a model speedup or a reproducible regression.
Both are retained, including the initially adverse result. All 576 actual
warmup/trial responses are exact across versions and matrices; verification
compares response fields, request hashes and workload names, not missing example
fields. No significant allocation change was measured. Ordinary process peak
RSS ranges overlap: baseline4,987,520–5,023,104 KiB; candidate4,985,216–5,022,592 KiB.

Separate normal-rate CPU and heap profiles, excluded from latency comparisons:

- Convolution cumulative samples: 230 ms before, 140 ms after.
- Packed GEMM microkernel remains dominant: 75.61% before, 75.48% after.
- Total sampled CPU: 26.57s before, 26.06s after. These single profiles are
  attribution evidence, not repeated performance measurements.
- `alloc_space`, `alloc_objects`, `inuse_space`, and `inuse_objects` reports retain
  the same major loading/packing consumers. Sampling does not prove zero small
  churn; the primitive's exact allocation tests and benchmarks provide that gate.

## Correctness and resource gates

- Focused primitive/convolution tests and races, count10: passed.
- Affected runtime, Qwen and MoJev package races: passed.
- Whole-tree NVIDIA-disabled CPU race (`-p=2 -count=1 -timeout=180s`): passed,
  explicit exit0, wall time5m06.16s for189 packages. Two earlier outer-tool
  interruptions stopped after113 package lines; those partial logs are retained
  and are not passes. The180s Go timeout is per package, not a whole-tree limit.
- `go vet ./...`, `go build ./...`, docs/layout checks: passed.
- New Go primitive/dispatch functions: 100% scoped statement coverage. This is
  not package-wide coverage and does not measure assembly.
- Linux ARM64 and RISC-V runtime/Qwen test cross-builds: passed. Disassembly
  confirms separate multiply/add instructions on amd64 and ARM64. ARM64 and
  RISC-V native execution remain unqualified.
- An ordinary grouped run was interrupted by the external100s tool timeout and
  is preserved as failed/incomplete evidence, not a pass.
- The subsequent candidate capacity512 grouped race passed in363.16s (Go test
  total364.542s, exit0). It validates the independent4096-total-token reference,
  hidden rows, changed/reordered concurrent requests and retained-output ownership.
  Max logits error `1.4007091522216797e-6`, changed logits
  `4.647299647331238e-7`, hidden `5.7220458984375e-5`; unchanged `3e-4` / `2e-3`
  gates. Post-concurrency heap was60,520 bytes below warm baseline. Race RSS
  peak11,151,676 KiB, zero swaps; not ordinary admission or a memory-saving claim.

Two delegated read-only review attempts timed out without findings. Source and
instruction inspection were completed locally; independent review remains a
limitation. No GPU execution occurred. The GPU hold, held-out quality/calibration,
hours-long retention and broader native-platform gates remain open;
`RuntimeReady=false`.

## Reproduction and evidence

Evidence directory: `/workspace/tmp/mojev-muladd-20260926`. Includes baseline and
candidate binaries, overlays, microbenchmarks, both complete timing matrices,
host observations, actual-response verification, profiles, disassembly, gate
logs and the interrupted ordinary-run log. Binaries are retained locally rather
than bundled into the downloadable evidence archive.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen \
  -run '^$' -bench '^BenchmarkBranchConvRow$' -benchmem -count=10
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 go test -race \
  ./backends/simd/runtime ./model/qwen \
  -run 'Test(VecMulAdd|BranchConvRow)' -count=10
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  GO_PHERENCE_MOJEV_GROUPED_BACKEND=simd \
  GO_PHERENCE_MOJEV_GROUPED_CAPACITY=512 \
  GO_PHERENCE_MOJEV_GROUPED_REPORT=OUTPUT.json \
  go test -race ./model/mojev -run '^TestReleasedGroupedTextScorer$' \
  -v -count=1 -timeout=900s
```

No decode, training or multimodal path changed; those workloads are outside this
slice. Loading and ordinary RSS are observed by the request probe, but unchanged
loading code was not subjected to a new cold-cache qualification. Further CPU
work should address the measured packed GEMM cost, not infer that convolution
now dominates or that the overall optimization plan is complete.
