# MoJev packed-only SIMD weights

The SIMD scorer can release about 1.99 GB of original projection weights after
packing when the caller drops its CPU scorer. Warm live heap falls from about
5.06 GB to 3.07 GB. The caller's scalar scorer remains usable, independent
numerical gates pass unchanged, and all four measured public responses are
exactly equal to the baseline. Peak loading RSS is unchanged.

## Kernel and ownership

`SgemmNTPackedOnlyTo` consumes the existing 16-column panel format without raw
row-major weights. It validates dimensions, strides, overflow, lengths and
output/input overlap before writing. Output columns must be a positive multiple
of 16. Complete row tiles use the existing assembly microkernel; row tails and
assembly-disabled execution use ascending-K scalar dot products from packed
panels. The existing raw/prepacked API and its callers remain unchanged.

The Qwen branch constructor validates the original model, copies only the layer
metadata and retains immutable non-projection norms/convolution weights.
Projection fields become private identity-only tensor handles mapped to owned
packed data. No original projection tensors or their backing arrays are retained.
Worker partitions always start on a complete 16-column boundary. The six-worker
cap and request-local scratch ownership are unchanged.

The MoJev host view keeps embeddings, head, final norm, RoPE, metadata and epsilon,
but no CPU encoder. Keeping the caller's original scorer alive still retains its
weights. Construction neither clears nor modifies any caller tensor. No GC calls
were added to inference or construction.

The scalar path and amd64 microkernel allocate zero per call in the tested
cases. ARM64 uses the existing microkernel and was cross-built, not executed.
The existing RISC-V microkernel allocates a temporary column buffer; its native
zero-allocation gate is explicitly skipped. Native foreign execution remains
unqualified.

## Memory and timing

Baseline: `ac40dc7f`, Linux amd64, Go 1.26.3, Intel i7-12700,
`GOMAXPROCS=6`, NVIDIA disabled. The approved checkpoint is hash-checked before
loading: weights SHA-256
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
The same configuration/tokenizer pins and four workloads from the
[GPU host probe](mojev-gpu-host-lifetime-20260926.md) are used. Each process runs
one warm-up and five calls per workload. Processes are serial; five baseline and
five changed processes were recorded, including interleaved before/after pairs.

The first memory pair, in bytes unless labelled:

| Measurement | Baseline | Packed-only |
|---|---:|---:|
| Live heap, both scorers | 5,124,001,576 | 5,124,157,824 |
| Live heap, accelerated scorer only | 5,124,001,576 | 3,133,636,488 |
| Live heap, after requests | 5,056,396,696 | 3,065,893,896 |
| Whole-process peak RSS, KiB | 6,049,888 | 6,050,656 |

The later pairs reproduced the retained-memory reduction. Sampled request RSS
fell from about 4.77–4.78 GiB to 2.91–2.93 GiB. Loading still temporarily holds raw
and packed projections together. After GC, the extra small metadata copies are
negligible beside the released payload. Probe phase `gpu_only` is a legacy label
from the shared harness; these runs use the SIMD backend.

Each process contributes its five-call median to benchstat:

| Workload | Baseline median ms | Packed-only median ms |
|---|---:|---:|
| Short, two choices | 282.929 | 244.723 |
| Short, eight choices | 335.543 | 349.639 |
| Two questions, two choices | 459.148 | 474.469 |
| Longer, two choices | 769.115 | 801.077 |

Benchstat found no statistically significant timing difference for any workload
at n=5. Several medians rose; these measurements do not establish a speedup or
exclude small regressions. Its 95% confidence intervals require at least six
samples. Benchstat's request allocation medians were 662/1000/1002/1593 before
and 657/999/1003/1591 after; variations include runtime and returned-output overhead.
There is no whole-request allocation reduction claim.

The packed-only microbenchmark has zero bytes and allocations per call on amd64.
It is slower than raw/prepacked when it must run scalar row tails; MoJev pads
projection rows to multiples of 12, avoiding those tails on amd64/ARM64. Even
complete-tile timing varied, so the memory saving is the accepted benefit.

## Accuracy, lifetime and concurrency

Native and assembly-disabled differential tests cover row tails, strides,
unaligned buffers, read-only operand overlap, output alias rejection, malformed
sizes/overflow, source preservation, output padding and exceptional scalar values.
Complete native tiles match the old prepacked API bit-for-bit. Warm projection
and worker tests retain zero-allocation assertions on the tested host.

`TestSIMDTextScorerHostLifetime` verifies caller scalar scoring after construction,
shared immutable host payloads, and collection of the original encoder root plus
sampled linear/full projection tensors and data while SIMD remains live. SIMD
base logits still match after GC. This does not enumerate every allocation;
retained-heap probes independently measure the payload reduction.

Completed released checks:

- Base/isolation/order/length cases: maximum logit error `3.69549e-6`, sampled
  hidden error `9.91821e-5`. Ordinary and race runs pass.
- Independent 512-token path/tree cases: maximum logit error `6.55651e-7` and
  hidden error `9.15527e-5`.
- Independent 4096-total grouped request at capacity 512: maximum logits
  `1.40071e-6`, changed candidate `4.64730e-7`, hidden `5.72205e-5`.
  Two simultaneous changed/reordered grouped requests pass ordinary and race.
- Grouped race completed in 485.58 seconds, process peak RSS 15,339,884 KiB;
  post-concurrency heap was 61,032 bytes below its warm baseline. That test
  intentionally keeps the CPU scorer alive, so it does not measure host release.

Gates remain `3e-4` logits and `2e-3` hidden. The first edited hidden-state test
kept borrowed internal scratch across later probes and failed. It now copies
the sample before reuse, retaining the independent expected values and existing
tolerances. One grouped race invocation was aborted without completion; its log
is preserved separately and the subsequent complete rerun passed.

The whole-tree NVIDIA-disabled race suite exits zero. Affected package races,
vet/build and ARM64/RISC-V builds and test-binary cross-builds pass. New kernel
functions and worker dispatch helpers reach 100% statement coverage; default
package coverage is SIMD runtime 81.8%, Qwen 47.8%, MoJev 67.0%, including opt-in
paths that default tests do not execute. Cross-builds are not native validation.

Two broader reviews timed out. Narrow read-only kernel/worker and constructor
ownership reviews completed with no concrete bug; the worker review identified
its required aligned-start invariant, which the scheduler guarantees and now
documents. No numerical approximation or tolerance change was introduced.

## Reproduction and scope

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race \
  ./backends/simd/runtime ./model/qwen ./model/mojev -count=3
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s

GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_MOJEV_SIMD=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev \
  -run '^(TestSIMDTextScorerHostLifetime|TestMoJevAcceleratedReleased)$' \
  -v -count=1 -timeout=600s

GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_GROUPED_BACKEND=simd GO_PHERENCE_MOJEV_GROUPED_CAPACITY=512 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestReleasedGroupedTextScorer$' \
  -v -count=1 -timeout=900s
```

Raw probes, benchmark samples, benchstat, coverage and gate logs are under
`/workspace/tmp/mojev-simd-host-20260926`. A post-commit 20-round reuse run on
`d99bd9d4` also passed in 814.33 seconds, including 512-path/4096-total requests,
eight concurrent callers, retained-output ownership and cancellation recovery.
Peak RSS was 6,094,556 KiB. Post-GC heap rose at most 124,104 bytes during retained
rounds and ended 67,944 bytes above the warm baseline after output release, with
two goroutines. The test intentionally retains the original CPU scorer. Logs and
snapshots are `retention-20.log` and `retention-20.json` in that evidence directory.
CI run `36229399479` passed. Native ARM64/RISC-V execution, hours-long service
retention and held-out quality/calibration remain open.
`RuntimeReady=false`.
