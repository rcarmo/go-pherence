# Native-path profiling, 2026-09-13

## Scope

These measurements cover checkpoint loading, a resident Qwen3 block, all 28
streamed decoder layers plus final norm, and go-264 reference-audio decoding.
They do **not** cover text-to-waveform generation: token embeddings, sampler and
learned audio codec are still missing. The stack starts with synthetic hidden
states, not accepted voice-conditioning tokens.

Hardware: Intel N100 VM, two vCPUs, ~5.8 GiB RAM, Go 1.26.2. Main benchmark: real
OmniVoice layer 0, 128 tokens, float32 compute from stored float16 weights.

## Measured results

| Case | Before | After |
|---|---:|---:|
| Resident block allocated bytes/call | 10,289,152 | 0 |
| Resident block allocations/call | 17 | 0 |
| Resident block time, sampled runs | 782.8 ms | 766.1 ms |
| 28-layer pass allocated bytes | 1,769,854,480 | 70,781,256 |
| 28-layer pass total time with profiling | 23.71 s | 21.57 s |
| 28-layer forward allocations | 0 after scratch reuse | 0 |

The stack before/after figures compare reusable activations with newly allocated
per-layer weights versus reusable activations **and** a reusable weight arena.
The process allocation counts were 829 vs 1,345: fewer bytes does not mean fewer
setup objects. The new count includes precomputed per-layer tensor-name strings.
These are allocated once, not in the conversion/forward hot paths. Timing is
noisy and not a controlled speedup claim. CPU profiles place ~86–89% of samples
in assembly `SgemmNT`; allocation removal cannot eliminate that compute cost.

After: weight arena 62,923,776 bytes; float scratch and position cache 7,734,784
bytes. These numbers exclude mmap checkpoint pages, Go/runtime overhead, output,
and the future codec. Stack sum was unchanged: 70117.40918272076.

## Changes

- `Block.NewWorkspace(tokens)` reserves one flat scratch arena.
- `Block.ForwardInto` reuses scratch, supports exact input/output alias for layer
  chaining, and has a zero-allocation regression test. No implicit growth.
- `Forward` remains the allocating convenience API; profiling uses `ForwardInto`.
- RoPE tables are cached by position and reused across Q/K and all heads. Rotation
  arithmetic uses existing vector dispatch. Norms operate in place where safe.
- Residual and mask adds use vector dispatch. Clear GEMM destinations before
  accumulation; all scratch is overwritten or cleared on each invocation.
- `Weights.NewLayerBuffer` precomputes names and slices. `LayerBuffer.Load`
  converts F16/F32/BF16 directly into the same float32 arena with zero successful
  call allocations. The canonical `half` package owns numerical conversions.
- Blocks referencing arena weights must finish before the next load. Workspaces
  and weight arenas are single-owner; concurrent inference needs separate ones.

## SIMD coverage

Assembly-backed projections, QK/PV GEMMs, RMSNorm, vector residual/mask/rotary
arithmetic and MLP multiplication use existing runtime dispatch. Head packing is
Go `copy`. Softmax exponentials and SiLU exponentials still use scalar `math.Exp`
on amd64. Weight conversion also remains scalar. No full-SIMD claim is made.
Profiles show these nonlinearities are a small fraction of present runtime;
approximate vector exponential work needs explicit numerical and voice-quality
validation rather than silently relaxing tolerances.

## Vulkan

`-mode capabilities -backend auto` distinguishes hardware from software Vulkan.
This VM has a loader/ICDs but no `/dev/dri`, and only llvmpipe is usable. Auto
selects CPU. Explicit Vulkan fails rather than pretending CPU fallback is GPU
execution. Hardware detection does not mean the OmniVoice Vulkan graph exists;
`implemented=false` and `native_dispatch=false` remain explicit.

Explicit CPU mode skips Vulkan initialization. Audio and checkpoint inspection
also skip compute selection. Auto probing is cached once per process and runs
inside short-lived CLI child processes, with a five-second timeout per attempt.
Hardware-only and software-allowed probes use separate children, so Vulkan's
global state, partial initialization leaks and allow-CPU policy do not affect the
parent. The CLI implements the private helper mode; generic library hosts that
do not implement that helper report no detected runtime rather than initializing
Vulkan themselves. These subprocesses only inspect devices, never run inference.

## Reproduce

```sh
make test-omnivoice vet-omnivoice build-omnivoice
bin/omnivoice -mode capabilities -backend auto
bin/omnivoice -mode block -backend cpu -model "$MODEL" -tokens 128 -iterations 5
bin/omnivoice -mode stack -backend cpu -model "$MODEL" -tokens 128 \
  -cpuprofile stack.cpu -memprofile stack.allocs
bin/omnivoice -mode audio -reference "$REFERENCE" -cpuprofile audio.cpu
GO_PHERENCE_REAL_OMNIVOICE="$MODEL" go test ./models/omnivoice -run '^$' \
  -bench BenchmarkRealBlockInto -benchmem -benchtime=5x
go tool pprof -top stack.cpu
go tool pprof -alloc_space -top stack.allocs
```

Profiles use exclusive-create files and refuse to overwrite existing paths.
CPU profiling spans the command. Allocation profiling uses rate 1 and perturbs
runtime, so use profile-free repeated runs for latency comparisons. CLI hot-path
allocation counters are process-wide snapshots; benchmark tests give the more
isolated allocation check.

## Validation

Tiny PyTorch fixtures still match every output within 9e-8 with and without a
blocked-key mask. Workspace reuse, in-place output, custom-position cache
invalidation, invalid sizes and zero-allocation calls are tested. Real layer 0
at 128 tokens: first-eight error 2.24e-7, peak error 9.85e-7, sum error 4.42e-5.
This real probe compares prefix and aggregates, not every full-stack element.
Full-waveform parity remains a later gate.

## Long-chunk reservation reuse (2026-09-13)

The same four-chunk, 13.42-second prepared-reference utterance now reserves inference
workspaces once at the largest planned shape. Smaller chunks use active views,
without padded attention. The old and new WAV files have identical SHA-256 hashes
(`bd1f3ec5a2bc5ece889b53d2462e0ab782274aa0b7953ebd0090b03ee90b79f9`).

The saved run took 335.17 seconds; the prior run took 344.46 seconds. This timing
pair is insufficient to establish a speedup. The `alloc_space` profile totals
368,551.84 KiB (359.9 MiB). Largest flat contributions include codec weight
conversion (82.2 MiB), the layer arena (60.0 MiB), decoder scratch (45.4 MiB), and
JSON slice growth (40.5 MiB). Tokenizer loading accounts for about 112.2 MiB
cumulatively. Post-processing and output assembly still allocate. Prepared
backbone/generation resizing and decoder calls retain zero-allocation tests.

Profile: `/workspace/tmp/omnivoice-long-reuse-v2.mem`; run metadata:
`/workspace/tmp/omnivoice-long-reuse-v2.json`. This cached-reference profile excludes
native reference encoding; it must not be compared directly with raw-reference
whole-run allocation totals.

## Tokenizer loading allocations (2026-09-13)

The isolated `BenchmarkOmniVoiceTokenizerLoad` measures the full real
`tokenizer.json` load, including inverse vocabulary and merge-rank construction.
It excludes token encoding and model inference.

| Loader | Allocated bytes/call | Allocations/call |
| --- | ---: | ---: |
| Before | 117,615,170 | 732,879 |
| Select merge representation before decoding | 86,414,285 | 581,456 |
| Also pre-size merge storage | 63,078,699 | 581,426 |

This reduces cumulative load allocations from 112.2 MiB to 60.2 MiB (46.4%).
The old loader attempted string decoding for array-form merges, allocating an
error for each entry before retrying. The loader now selects the format from the
first entry and reserves storage with a non-allocating pass over already validated
JSON. The standard JSON decoder still performs decoding and validation. Array
merges go directly into their final storage; string merges use `strings.Cut`.

Baseline timings ranged from 363 to 379 ms. Final five-iteration runs ranged
from 307 to 329 ms; other intermediate runs were substantially noisier. These
measurements do not establish an end-to-end synthesis speedup or a new whole-run
allocation total. The previous long-run profile predates this loader change.

Reproduce with:

```sh
GO_PHERENCE_REAL_OMNIVOICE=/path/to/OmniVoice go test ./loader/tokenizer \
  -run TestOmniVoiceRealTokenizerParity -bench BenchmarkOmniVoiceTokenizerLoad \
  -benchtime=5x -count=3
```

All 24 real-tokenizer fixtures pass. Tests cover both merge representations,
empty/null merges, whitespace, escaped strings, nested JSON sizing and malformed
mixed arrays. The sizing helper also passed a 10-second fuzz run (321,269
executions). A focused review found no new correctness issue. Affected-package
tests, vet, race/no-CGo tests and the ARM64 CLI cross-build pass.

## Exact amd64 SIMD panel packing (2026-09-13)

The full native CPU profile attributed 7.65 seconds (8.90%) to scalar B-panel
packing. The amd64 packer now transposes four columns from each of sixteen rows
with AVX loads/shuffles/stores. It performs no floating-point arithmetic and
preserves all input bits. Full panels use SIMD; partial panels retain scalar
zero-fill and the final one to three columns use bounded scalar loads. Dispatch
requires the same AVX2/FMA runtime gate as the GEMM backend.

| K | Scalar panel | SIMD panel | Allocations |
| --- | ---: | ---: | ---: |
| 128 | 1.79–1.81 µs | 0.228 µs | 0 |
| 1024 | 14.64–15.49 µs | 3.87–3.97 µs | 0 |
| 3072 | 41.82–42.05 µs | 11.88–11.89 µs | 0 |

Benchmark: `go test ./backends/simd/runtime -run '^$' -bench '^BenchmarkPackBNT$'`.
The direct speedup ranges from 3.5× to 7.9×. These hot-cache panel measurements
are distinct from full-model performance.

The same raw-reference native synthesis run produced a byte-identical three-second
WAV: SHA-256 `560fe8ae07d6af712f51f64f553d8e8a0b6146778d31a3a944a6db90b61a1e0e`.
Packing consumed 3.58 seconds (4.22%) in the new CPU profile. Whole-command time
was 85.78 seconds versus 85.95 seconds previously, so no whole-model speedup is
established. Generation took 69.55 seconds; reference encoding took 9.91 seconds.
The full allocation profile totals 1,152,269.29 KiB (1,125.3 MiB), including native
reference encoding and the preceding tokenizer-load optimisation. The packer
itself adds no allocations.

Artifacts: `/workspace/tmp/omnivoice-full-native-pack-v7.{cpu,mem,json}` and
`/workspace/tmp/synthetic-spock-full-native-pack-v7.wav`.

Validation includes bitwise layouts with signed zero, NaN payloads, subnormals,
strides, offsets and tails; GEMM boundary shapes and nonzero output accumulation;
Linux/amd64 PROT_NONE guard pages immediately after each independent source row
and destination; race/no-CGo tests; CPU-feature-disabled fallback tests; and ARM64
cross-builds. Review found no new correctness issue. OmniVoice vet passes; direct
SIMD-runtime vet reports existing `q8dot_amd64.s` return-offset warnings in untouched
code. Full-repository builds retain the known SpacemiT/DiffusionGemma failures.

An alternating-broadcast scheduling experiment for the 6×16 GEMM microkernel
showed no consistent throughput improvement and was not enabled. Representative
finite inputs are required for throughput benchmarks: subnormal test values caused
FP assists to dominate the first experiment. Correctness tests still include them.
