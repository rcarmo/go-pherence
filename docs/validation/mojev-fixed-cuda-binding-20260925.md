# MoJev fixed CUDA launch binding

A fixed-signature CUDA launch binding removes reflective marshalling from the
NVIDIA hot path. It follows [launch batching](mojev-launch-batch-20260925.md)
at `85fff290` (CI `36198115627` passed).

On Linux amd64/arm64, runtime initialisation resolves `cuLaunchKernel` and wraps
its integer/pointer-only signature with the existing `purego.SyscallN` API. It
reuses an eleven-word argument array under `cudaMu`. Other architectures retain
the typed purego binding; an unresolved symbol leaves the existing binding
untouched. No PTX arithmetic or model execution order changed.

The wrapper holds GC-visible `unsafe.Pointer` roots until CUDA returns, then
clears both pointers and argument words. Escape analysis confirms incoming host
tables escape before conversion to `uintptr`. All runtime launch entry points
hold the same mutex and OS-thread pin. The plain launch wrapper now explicitly
keeps its typed argument slice alive, matching the existing stream wrapper.
This shared binding remains non-reentrant; it must only run inside a driver
scope.

## Measurements

Same i7-12700, RTX 3060, driver 580.173.02, Go 1.26.3 and `GOMAXPROCS=6` as the
preceding pass. Checkpoint SHA-256 is
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`, revision
`0c8695b6252f4205907433d4e196a94f032e60c3`. The existing approved checkpoint was
used, one model process at a time. Each full request includes JSON parsing,
tokenisation, inference and response marshalling. Preparation and hash checking
are outside the timed region. One warmup precedes five samples; entries below
are medians.

| Request | Prior ms | Fixed binding ms | Allocations prior → fixed | Bytes prior → fixed |
|---|---:|---:|---:|---:|
| Two choices | 91.32 | 89.16 | 9,431 → 697 | 327,136 → 54,000 |
| Eight choices | 367.18 | 364.42 | 36,193 → 1,260 | 1,261,720 → 170,152 |
| Two questions | 177.85 | 180.17 | 18,568 → 1,100 | 643,800 → 97,528 |
| Longer context | 184.36 | 184.24 | 10,373 → 1,639 | 367,232 → 94,080 |

Objects fall 84–97%; allocated bytes fall 74–87%. Latency is essentially
unchanged, and no general speedup is established. All four saved responses
match the prior run. Device buffers and weight residency are unchanged; the
binding adds one small, fixed host allocation at initialisation.

An initial `SyscallN` implementation still allocated one variadic argument
slice per launch. The follow-up profile exposed that cost; reusing the words
removed it. A native empty-kernel test measures **3 allocations per 256-launch
batch**, versus **2,819** for the reflective binding. The three remaining objects
come from context setup, so the full batch is not allocation-free.

Under the race detector, purego's `sync.Pool` entries are randomly dropped and
allocation counts rise. The normal test enforces at most four objects per batch;
the race test checks the reduction against the reflective baseline. It does not
apply an ordinary-build allocation threshold to instrumented execution.

## Verification

- Synthetic ABI differential against `purego.RegisterFunc`: all eleven arguments,
  high-bit dimensions and pointer values, stack-passed slots, error return and
  forced GC during callback. Native amd64 pass; ARM64 is cross-built only.
- Native CUDA batch test, ordinary and race, three repetitions: allocation
  bounds, concurrent batches, diagnostic launch checking and cleanup pass.
- Released MoJev GPU oracle/isolation/race: eight cases, max logit error
  `4.7087669e-5`, max hidden error `7.8797340e-4`; fixed gates remain `3e-4` and
  `2e-3` respectively.
- Whole-tree CPU race suite exits 0; vet, build and ARM64/RISC-V cross-builds pass.
- Direct/batched kernel memcheck: zero errors; racecheck: zero hazards.
- Independent focused review found the missing explicit `KeepAlive` at the plain
  wrapper; that wrapper now mirrors the stream path. No other binding issue was
  found in the scoped review.
- The ABI-only coverage run covers the fixed wrapper at 100%; binding resolution
  and native execution are tested separately. No repository-wide coverage claim.

Commands:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./backends/nvidia/runtime -count=3
GO_PHERENCE_TEST_CUDA_LAUNCH=1 GOMAXPROCS=6 \
  go test ./backends/nvidia/runtime -run '^TestFixedCUDAKernelLauncher' -v -count=3
GO_PHERENCE_TEST_CUDA_LAUNCH=1 GOMAXPROCS=6 \
  go test -race ./backends/nvidia/runtime -run '^TestFixedCUDAKernelLauncher' -v -count=3
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s
```

Raw samples, executable probes, profiles and logs are preserved in
`/workspace/tmp/mojev-fixed-launch-20260925`; the prior binary/evidence is in
`/workspace/tmp/mojev-optim-20260925`. Peak RSS and long-lived retained-heap
qualification were not rerun for this small binding change.

The new allocation profile is led by tokenisation; launch marshalling is no
longer a hotspot. Ancestor reuse, controlled CPU scheduling measurements,
cancellation, admission and task-quality validation remain separate open work.
`RuntimeReady` stays false.
