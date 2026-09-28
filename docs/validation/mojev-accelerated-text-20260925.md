# MoJev text SIMD and PTX: allocation pass

The isolated F32 text scorer has explicit CPU SIMD and NVIDIA PTX backends.
The CPU reference remains available. The initial acceleration implementation is
`9956264a`; this follow-up reduces allocation churn and tightens validation and
cleanup. `RuntimeReady=false`: images, held-out task quality, cancellation,
full-context admission and service deployment still need qualification.

## Execution

`NewSIMDTextScorer(cpu, maxTokens)` packs immutable projection weights once.
It batches branch tokens through the existing assembly SGEMM kernels and uses
assembly-dispatched dot, SAXPY and vector addition for attention, recurrence and
residuals. Unsupported CPU targets retain the checked scalar fallbacks. No new
approximate exponential or lower-precision mode is enabled.

Scratch belongs to each scorer and a mutex serialises calls. Up to six workers
live for one branch, then join; no idle goroutines keep discarded models alive.
Workers are reused across projections within the branch. Full-attention scores,
output rows and embedding views use bounded reusable storage. `ForwardInto`
checks destination geometry and input/RoPE overlap and copies results only after
successful execution. Public results remain owned. Text requests are validated
and packed once, rather than tokenised again for validation and final assembly.

`NewNVIDIATextScorer(cpu, maxTokens)` keeps F32 encoder weights and scratch on the
GPU. It requires compute capability 8.6 or later and 4 GiB free headroom before
construction. The CPU provides embeddings and scorer-head readout. Both
constructors accept capacities from 3 to 512 tokens per candidate path; that
bound is smaller than the reference request limit of 4096 packed tokens.
Neither backend shares ancestors between candidates yet.

The GPU constructor now validates fixed layer geometry and dense finite weights
before allocating device buffers. `Close() error` serialises against active
scoring. A module synchronisation/unload failure disables inference but retains
resources for a retry. Check the error before shutting down the NVIDIA runtime.
If construction and cleanup both fail, the constructor returns the error and a
non-nil closed scorer solely so the caller can retry `Close`.

## Measurements

Host: Intel i7-12700, Linux/amd64, Go 1.26.3, `GOMAXPROCS=6`, RTX 3060,
driver 580.173.02. Model revision
`0c8695b6252f4205907433d4e196a94f032e60c3`; safetensors SHA-256
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
Only the approved existing checkpoint was used, one model process at a time.

Each request starts from identical fresh JSON. Timings include parsing,
tokenisation, inference, answer assembly and JSON output; they exclude HTTP,
telemetry, checkpoint hash verification and preparation. One warmup precedes
five samples. The before binary includes the validation fixes but retains the
initial SIMD execution. Both binaries and their harness are preserved outside
the repository in `/workspace/tmp/mojev-optim-20260925`.

| Request | SIMD before ms | SIMD after ms | PTX after ms | SIMD bytes/request before → after | SIMD allocs/request before → after |
|---|---:|---:|---:|---:|---:|
| Two choices | 529.36 | 597.21 | 90.15 | 5,064,552 → 61,264 | 6,793 → 687 |
| Eight choices | 2599.77 | 2556.21 | 366.57 | 25,629,648 → 185,768 | 24,637 → 1,190 |
| Two questions | 945.15 | 884.94 | 177.17 | 9,613,848 → 103,720 | 12,714 → 1,062 |
| Longer context | 2217.25 | 1495.62 | 184.95 | 17,487,360 → 95,824 | 12,221 → 1,616 |

Values are medians. Short-request latency is noisy: the final two-choice samples
range from 455 to 676 ms, and an earlier after run had a 504 ms median. The final
two-choice median regresses by 12.8%; this pass has no general latency acceptance.
The longer-context reduction repeated across three after runs. Allocation counts
fell 87–95% and bytes fell 98.8–99.5% in these workloads. No significance result
is available: the installed `benchstat` build failed to finish on the small
converted input files within a bounded run. Raw samples are retained.

A later publication-check run, after restoring pre-inference label validation,
measured 654.80/3055.16/1143.33/1870.35 ms for the same four requests. Its median
allocation counts were 688/1170/1067/1624 and bytes were
92,816/179,912/106,344/96,720. This spread reinforces the need for a controlled
paired latency experiment. The label check adds no measured allocation trend;
latency acceptance stays open.

Peak process RSS was 6,092,192 KiB before, 6,062,956 KiB after SIMD and
6,145,180 KiB for PTX. The PTX scorer reported 2,024,992,000 resident device bytes
at capacity 256. Host loading and weight packing dominate memory; reducing warm
allocations does not remove those copies. Observed preparation times were
4.82/4.67 seconds for SIMD before/after and 6.99 seconds for PTX. Post-GC,
long-running retained-memory admission needs a separate experiment.

## Profiles and regression checks

Before this pass, projection dispatch accounted for 64.8% of objects in the
bounded allocation-attribution run and scalar grouped attention another 14.6%.
Reusable workers, assembly attention and scratch eliminate those repeated
allocations. Tokenisation is now the largest warm allocation source. The final
CPU profile assigns 75.25% of samples to the assembly matrix microkernel;
projection throughput and repeated candidate ancestors are the next targets.

Warm projection and attention tests assert zero allocations with preallocated
buffers. Projection tests cover padded and unpadded rows, row residues 1–13,
parallel output tails 511/512/513, guards, source preservation and disabled
assembly dispatch. The released encoded-SIMD test enforces a ceiling of 140
allocations for the fixed four-candidate fixture, separate from public JSON and
tokeniser work.

The eight pinned F32 cases pass unchanged tolerances of `3e-4` for logits and
`2e-3` for hidden rows. Maximum observed errors:

| Backend | Logits | Hidden rows |
|---|---:|---:|
| SIMD | 4.6759844e-5 | 7.7509880e-4 |
| PTX | 4.7087669e-5 | 7.8797340e-4 |

Tests cover exact sibling/question isolation, token and length substitutions,
permutations, repeatability, four concurrent callers, owned output, transactional
Into failures, invalid tokens/capacity, close racing with scoring, and mocked
PTX module sync/unload failure retries. The original CPU reference is unchanged
apart from sharing the single-pack public request assembly.

Verified commands include:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s
GOMAXPROCS=6 GO_PHERENCE_MOJEV_SIMD=1 GO_PHERENCE_MOJEV_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestMoJevAcceleratedReleased$' -count=1 -v
GO_PHERENCE_DISABLE_NVIDIA=1 go vet ./...
GO_PHERENCE_DISABLE_NVIDIA=1 go build ./...
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build ./model/mojev ./model/qwen
GOOS=linux GOARCH=riscv64 CGO_ENABLED=0 go build ./model/mojev ./model/qwen
make docs-check
```

The whole-tree race log ends with exit 0. Cross-builds do not qualify native
ARM64/RVV performance. Default model-free package coverage is 69.3% for MoJev,
46.8% for Qwen and 18.1% for NVIDIA runtime; hardware/checkpoint execution is
opt-in. The released coverage run exercises the accelerated success paths but
still leaves driver failures and several validation branches uncovered. The
repository's 90% changed-code target is not closed by this pass.

An independent review found the constructor and close-ownership gaps addressed
here. The follow-up review timed out; final independent approval is outstanding.
The PTX kernels are unchanged in this pass. Earlier kernel memcheck/racecheck
runs found zero errors/hazards. Nsight Systems still produced no CUDA kernel
records on this host, so no GPU kernel-time breakdown is available.

## Remaining work

- Resolve noisy short-request CPU measurements and assess projection scheduling.
- Reuse state/question ancestors with exact isolation and numerical checks.
- Reduce candidate/head boundary and GPU launch allocation overhead.
- Obtain usable GPU timing attribution before further PTX kernel changes.
- Finish lifecycle/failure coverage, cancellation and maximum-capacity admission.
- Validate task quality/calibration and native non-amd64 hardware separately.
