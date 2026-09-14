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

## SIMD sine and codec Snake (2026-09-13)

`SinF32To` uses AVX2/FMA for finite float32 phases with |x| <= 32. It reduces
by a split pi/2 constant and evaluates degree-11 sine / degree-10 cosine
polynomials, then selects the quadrant. Out-of-range and exceptional values,
negative zero and short tails use scalar `math.Sin`. Other architectures use
scalar sine. Partial overlap is rejected; exact in-place use is supported.

Dense [-32,32] and 200,000 seeded random tests observed maximum absolute error
5.96e-8; the acceptance threshold is 2e-6. Exact float32 quadrant neighbours and
SIMD-range boundaries are also tested. This is an approximation, not bitwise
math.Sin equivalence.

Codec encoder/decoder Snake now uses 256-float stack scratch and separate vector
scale, square and add operations. These retain float32 expression boundaries
without adding workspace allocations. A 4,096-value channel benchmark measures
53.98–54.63 microseconds scalar versus 15.86–16.23 microseconds SIMD (about 3.4×),
with zero allocations. The direct 1,024-value sine benchmark measured about 7.5×.

Real-checkpoint encoder parity passes all 8×3 codes. The real decoder matches
all 1,920 reference samples with maximum absolute error 5.74e-7 and zero prepared
allocations. Native preprocessed reference encoding still produces the same 832
codes as the previous checkpoint.

Full raw-reference synthesis took 92.74 seconds versus 85.78 in the preceding
run. Reference encoding fell from 9.91 to 8.30 seconds and decode/save from 4.07
to 2.98 seconds, while generation rose from 69.55 to 81.06 seconds. This single
pair does not establish a whole-model speedup. No backbone math changed.

The resulting three-second WAV differs in 61 of 72,000 PCM16 samples; every
difference is one integer step, with RMS delta 0.0291 PCM16 units. It is no longer
byte-identical. New SHA-256:
`1d8908e82a5af5c6c1cd191d94a32d4cf8ce1c811b420dcea009c726b2771ea5`.
Artifacts: `/workspace/tmp/omnivoice-full-native-sine-v8.{cpu,json}`, waveform
`/workspace/tmp/synthetic-spock-full-native-sine-v8.wav`, and reference codes
`/workspace/tmp/omnivoice-reference-sine-v8.json`.

Tests cover overlap, tails, signed zero, exceptional values, exact quadrant
neighbours, mixed fallback phases across 256-element Snake tiles and allocation
counts. Runtime/model tests, race/no-CGo checks, disabled-AVX2/FMA fallback,
OmniVoice vet and ARM64 cross-builds pass. Full-repository build failures remain
in the previously identified unrelated packages.

Simple K-chunk calls around the current packed GEMM API would store partial sums
and change accumulation order. Cache blocking needs an accumulator-preserving
kernel/epilogue design before exact-parity deployment; no K-blocking change was
implemented or benchmarked in this stage.

## SIMD SiLU finishing stage (2026-09-13)

On AVX2/FMA amd64 hosts, `SiLUMulExpTo` now finishes with separate vector
add/divide/multiply instructions. It uses hardware division, not a reciprocal
estimate. The float32 rounding steps match the previous scalar division plus
vector multiplication. Exact destination aliasing with gate/up is supported;
partial overlaps and scratch overlap remain rejected. Other architectures and
short tails use scalar finishing. Scratch is temporary and its final contents
are not part of the API contract.

The existing 128×3072 FFN benchmark fell from 4.441–4.448 ms to 3.908–4.006 ms
(roughly 10–12%), with zero allocations. The generic SiLU baseline remains around
6.08 ms. Logs: `/workspace/tmp/omnivoice-silu-div-{before,after}.log`.

Full native synthesis took 78.38 seconds (66.90 generation, 3.12 decode/save),
compared with 92.74 seconds in the preceding sine-enabled run. Run-to-run variance
prevents attributing that whole difference to this change. The WAV is byte-identical
to the sine-enabled baseline, SHA-256
`1d8908e82a5af5c6c1cd191d94a32d4cf8ce1c811b420dcea009c726b2771ea5`.
Artifacts: `/workspace/tmp/omnivoice-full-native-div-v9.{cpu,json}` and
`/workspace/tmp/synthetic-spock-full-native-div-v9.wav`.

Direct bitwise tests compare scalar and SIMD finishing across vector/tail sizes
and exact aliases, including finite extremes, subnormals, signed zero, infinities
and NaN classification. Contract tests verify rejection without mutation. Runtime
and model tests, race/no-CGo checks, disabled-AVX2/FMA fallback, OmniVoice vet and
ARM64 cross-builds pass. No new waveform attachment is needed for identical audio.

Encoder ELU still uses scalar `float32(exp(float64(x))-1)` for negative inputs.
Replacing it with float32 exp followed by subtraction loses small negative values
near zero; it needs a separate near-zero-safe implementation and parity tests.

## Encoder SIMD ELU (2026-09-13)

`ELUF32To` uses an AVX2/FMA expm1 polynomial for negative inputs in
[-16, -2^-20). Range reduction uses a split ln(2); reconstruction avoids rounding
exp to float32 before subtracting one. Smaller negative magnitudes remain scalar
to preserve the existing float64 exp/subtract cancellation. Positive SIMD lanes
are masked to zero during polynomial evaluation and copied back bitwise. Signed
zero, positive infinity and NaN payloads pass through unchanged. Out-of-range
negative values, unsupported CPUs and short tails use the scalar reference.

Dense/random tests observed maximum absolute error 5.96e-8 and relative error
1.19e-7 over the tested regular range; the absolute acceptance limit is 2e-6.
Tests cover tiny-negative powers and ULP neighbours, exact tiny-region results,
positive extremes in mixed SIMD lanes, NaN payloads, aliasing and tails.

For 4,096 representative mixed-sign inputs, scalar ELU measured 34.95–36.58 us
and dispatch 10.85–11.04 us (about 3.2×), with zero allocations. All-negative
1,024-input cases measured about 5.5× faster. A deliberately fallback-heavy mixed
benchmark remained slower than scalar (about 12.5 versus 10.5 us); this kernel is
not uniformly faster for arbitrary distributions.

The semantic codec encoder now calls this kernel. Real-checkpoint feature
encoding matches all 8×3 codes with zero prepared allocations. Native raw reference
encoding preserves all 896 codes; preprocessed reference encoding preserves all
832 codes. Artifacts: `/workspace/tmp/omnivoice-reference-elu-{raw-v10,v10}.json`.
No backbone/decoder math changed and no full synthesis timing was repeated for
this stage. Reference-code equality is the integration gate here, not a new
waveform or throughput claim.

Runtime/model tests, race/no-CGo checks, disabled-AVX2/FMA fallback, OmniVoice vet
and ARM64 CLI cross-build pass. Focused review found and prompted a fix for NaN
payload preservation; explicit tests now cover positive/negative quiet and
signalling NaNs. Full-repository build failures remain unrelated to these changes.

## HuBERT SIMD erf-GELU (2026-09-13)

`GELUErfF32To` approximates the erf formula used by HuBERT; it does not substitute
tanh-GELU. The AVX2/FMA path uses Abramowitz–Stegun 7.1.26 for erf and the existing
bounded exp reduction/polynomial pattern. Its window is 2^-12 <= |x| <= 8.
Tiny/exceptional values, out-of-range inputs, short tails and unsupported CPUs
retain scalar `math.Erf`. Exact in-place operation is supported; partial overlap
is rejected. No scratch allocation is required.

Dense/random tests observed maximum absolute GELU error 4.77e-7, below the 2e-6
acceptance limit. Tests also cover window boundaries, signed zero, exceptional
values and float32 neighbours around internal exp range-reduction transitions.

| Inputs | Scalar | Dispatch | Allocations |
| --- | ---: | ---: | ---: |
| 1024, eligible | 63.10 us | 6.40 us | 0 |
| 3072, eligible | 189.78 us | 18.04 us | 0 |
| 1024, fallback-heavy | 29.90 us | 30.50 us | 0 |
| 3072, fallback-heavy | 90.81 us | 100.99 us | 0 |

The roughly 10× direct gain applies to eligible runs; fallback-heavy input can be
slower. Log: `/workspace/tmp/omnivoice-gelu-benchmark.log`.

All HuBERT GELU sites now use the kernel. Real-checkpoint parity passes at five
frames (max 3.81e-6, RMS 4.40e-7) and 100 frames (max 9.51e-6, RMS 3.66e-7).
Prepared `ExtractInto` remains allocation-free. Native raw and preprocessed
reference codes remain unchanged: 896 and 832 codes respectively. Artifacts:
`/workspace/tmp/omnivoice-reference-gelu-raw-v11.json`,
`/workspace/tmp/omnivoice-reference-gelu-v11.json`, and
`/workspace/tmp/omnivoice-gelu-hubert-parity.log`.

No backbone or decoder math changed; no full synthesis timing was repeated.
Runtime/model tests, race/no-CGo checks, CPU-feature-disabled fallback, OmniVoice
vet and ARM64 test/CLI cross-builds pass. Focused review found no functional issue.
Whole-repository builds still fail in the previously identified unrelated packages.

## Sampler SIMD log-softmax (2026-09-13)

Guidance log-softmax now reuses `vocabScratch2` for float32 max-shifted logits and
bounded SIMD exponentials. Summation remains float64 and sequential; the final
log normaliser uses the previous formula. Rows containing NaN or positive infinity,
and all-negative-infinity rows, use the original scalar policy. Mixed finite/-Inf
rows safely map masked entries to zero probability. No workspace was added.

This approximates the former float64 exponential path. Near-tied confidence scores
can change ordering on other inputs; deterministic seeds still reproduce this
implementation, not the former scalar path or PyTorch RNG.

A 1,025-class benchmark improved from 17.4–17.7 us to 6.1–6.4 us (about 2.8×),
with zero allocations. Upstream guidance fixtures pass their existing tolerance,
as do deterministic greedy generation and full guidance/prediction/selection
allocation tests. Additional tests cover extreme finite values, exceptional rows,
vector/tail lengths and mid-generation cancellation followed by deterministic reuse.

The full raw-reference native sample is byte-identical to the preceding waveform,
SHA-256 `1d8908e82a5af5c6c1cd191d94a32d4cf8ce1c811b420dcea009c726b2771ea5`.
Whole-command time was 86.76 seconds (75.10 generation, 3.25 decode/save), compared
with 78.38 seconds in an earlier run. Timing variation prevents a whole-model
speedup claim. Artifacts: `/workspace/tmp/omnivoice-full-native-sampler-v12.{cpu,json}`
and `/workspace/tmp/synthetic-spock-full-native-sampler-v12.wav`.

Runtime/model tests, race/no-CGo checks, disabled-AVX2/FMA fallback, OmniVoice vet
and ARM64 CLI cross-build pass. Review found no scratch-use or exceptional-row bug;
its requested mid-generation cancellation/reuse coverage was added. Full-repository
build failures remain in unrelated packages.

## Exact K-blocking experiment and fallback audit (2026-09-13)

The latest full profile (`omnivoice-full-native-sampler-v12.cpu`) attributes 62.46%
of sampled CPU time to the 6×16 GEMM microkernel. A continuation-kernel experiment
stored raw accumulators between K tiles, preserving each element's FMA order and
applying alpha/C only once. Bitwise tests covered K tails, strides and nonzero C.
The candidate allocated nothing after caller scratch setup.

It did not improve throughput on this N100:

| Shape / K tile | Existing full-K | Exact K-blocked |
| --- | ---: | ---: |
| 126×1024×1024 / 128 | 6.16–6.21 ms | 6.42–6.98 ms |
| 126×1024×1024 / 256 | 6.19–6.42 ms | 6.34–6.43 ms |
| 126×3072×1024 / 128 | 19.45–19.47 ms | 20.12–20.54 ms |
| 126×3072×1024 / 256 | 19.69–19.81 ms | 19.92–20.66 ms |

For M=126 and K=1024, candidate scratch was 15.9/23.9 KiB at K tiles 128/256,
versus 64 KiB packed scratch in the existing kernel. That local saving did not
justify enabling a neutral/slower path. The candidate was removed from runtime
source and archived at `/workspace/tmp/omnivoice-kblocked-experiment/`; measurements
are in `/workspace/tmp/omnivoice-kblocked-benchmark.log`. This does not rule out
other tiling, vector widths or parallel strategies; these tested variants failed
the throughput gate.

`BenchmarkPackedModelProjections` now provides ordinary finite-input baselines
for representative projections, including row tails. It avoids subnormal-heavy
inputs that previously obscured throughput with FP assists.

The backend report now exposes `approximate_nonlinear_simd` and `scalar_fallbacks`.
`full_graph_simd` remains false: reductions, selection, scatter/packing orchestration,
preprocessing and range/architecture fallback paths still contain scalar work.
The current host reports AVX2/FMA nonlinear SIMD, no hardware Vulkan device and
software llvmpipe only. Auto selects CPU; explicit Vulkan execution is unsupported.
Artifact: `/workspace/tmp/omnivoice-capabilities-audit.json`.

No inference kernel was changed in this audit stage. Runtime/model tests, race,
no-CGo, CPU-feature-disabled reporting and ARM64 CLI cross-builds pass.

## Long and Portuguese end-to-end checks (2026-09-13)

| Sample | Audio | Command time | Real-time factor | Clipped PCM16 samples |
| --- | ---: | ---: | ---: | ---: |
| English, four chunks | 13.42 s | 341.48 s | 25.4× | 0 |
| Portuguese, one utterance | 4.13 s | 114.22 s | 27.7× | 0 |

Both runs use cached reference codes and eight steps; reference encoding is not
included in these timings. English chunk times were 78.02, 77.51, 84.08 and 101.33
seconds. The English waveform differs from the prior long baseline by one PCM16
step at 307 positions out of 322,080; RMS delta is 0.03087 integer units. Whole-run
throughput did not improve conclusively. ASR recovered the expected words in
English and Portuguese; listening quality remains unaccepted. See
[IMPLEMENTATION.md](IMPLEMENTATION.md) for settings, evidence and artifact paths.

## Opt-in resident float32 layers

A matched eight-step, 75-frame cached-reference run produced identical WAVs in
streamed and resident modes (SHA-256
`e2c01684385c2b561d4b086f1ba23fdfb7c9cf64dc227f32df91edb7f665d579`).

| Mode | Total | Generation including cache setup | Cache setup | Extra decoder arenas |
| --- | ---: | ---: | ---: | ---: |
| Streamed | 69.99 s | 66.62 s | disabled | 0 |
| Resident | 68.69 s | 64.96 s | 4.10 s | 1,761,865,728 bytes |

This single pair shows only a modest total-time difference; it does not establish
a reliable whole-model speedup. Reuse across requests should amortise construction,
but persistent serving is not yet implemented. Peak RSS was not measured.

Artifacts: `/workspace/tmp/omnivoice-resident-v1.{cpu,json}` and
`/workspace/tmp/omnivoice-streamed-compare-v1.json`. Tiny-fixture exact parity,
shared CFG cache, resizing, budget accounting, partial-build cancellation,
zero-forward-allocation, race/no-CGo tests, CLI validation and ARM64 builds pass.
Resident mode does not yet eliminate weight packing or cache embeddings/heads.

## Resident prepacked SIMD comparison

Matched cached-reference, eight-step, 75-frame synthesis:

| Mode | Total | Cache setup | Decoder cache | Sampled process peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Raw resident | 71.32 s | 1.87 s | 1,761,865,728 B | 2,786,300 KiB |
| Resident + prepacked | 83.01 s | 7.47 s | 3,523,473,408 B | 4,118,236 KiB |

RSS high-water marks were polled from `/proc/PID/status`; this is process resident
memory, not the cache budget. The prepacked process reported 4,480 KiB of swap;
the raw-resident comparison reported none. Both WAVs match the streamed baseline
SHA-256 `e2c01684385c2b561d4b086f1ba23fdfb7c9cf64dc227f32df91edb7f665d579`.

Direct 1024×1024 projection benchmarks were mixed: prepacking slightly helped
126-row shapes and hurt 128-row shapes in these runs. Retaining panels eliminates
packing per call, but does not guarantee faster execution when cache residency,
row tails and memory pressure change. Prepacking stays opt-in; combined worker
and persistent-serving measurements are still required.

Artifacts: `/workspace/tmp/omnivoice-prepacked-v1.{json,rss}`,
`/workspace/tmp/omnivoice-resident-compare-v2.{json,rss}` and
`/workspace/tmp/omnivoice-prepacked-benchmark.log`.

Tests cover bitwise GEMM parity, strides/tails, overlapping-buffer rejection,
zero allocations, raw-to-packed transactional upgrades and sibling ownership.
Model/runtime race/no-CGo, disabled-AVX2/FMA fallback, OmniVoice vet and ARM64
cross-build checks pass. The CLI refuses prepacking without active SIMD GEMM.

## Request-local embeddings and sparse target heads (2026-09-14)

Generation now computes raw text/reference prefix embeddings once per request.
Each positive denoising step copies that prefix and embeds the changing target.
The full non-causal transformer still executes for both CFG branches. Prefix
buffers are owned by Generation, rebuilt on every invocation, and reserved for
up to `maxTokens-1` positions per branch to support target shrink/restore.
Their float32 payload is `4*hidden*(conditional.maxTokens-1)` bytes plus the
corresponding unconditional reservation when guidance is enabled.

Head projection omits full SIMD row groups with no masked target time remaining
across any codebook. Adjacent active groups are merged; head weight chunking and
the original full-sequence row/tail grouping are unchanged. Revealed logits can
remain stale: the dense sampler still consumes the same RNG draws and excludes
those rows during confidence selection. Public full/target-forward APIs retain
their dense output contract.

Matched 75-frame, eight-step, two-thread/two-worker, resident-float32 synthesis
of “The evidence is insufficient, Captain.” against parent `17052eed`:

| Trial | Before wall seconds | After wall seconds |
| --- | ---: | ---: |
| First | 60.250 | 54.419 |
| Repeat | 51.282 | 56.784 |
| Mean | 55.766 | 55.602 |

The mean difference is only 0.3%, with opposite directions in the pairs; no
end-to-end speedup is demonstrated. Peak RSS stayed near 2,786,000 KiB. All four
WAVs have SHA-256
`e2c01684385c2b561d4b086f1ba23fdfb7c9cf64dc227f32df91edb7f665d579`.
Local measurement records: `/workspace/tmp/omnivoice-prefix-{before,after}*.json`.
Command (use the parent binary for before; unique output paths are required):

```sh
bin/omnivoice -mode synthesize -backend cpu \
  -model /workspace/projects/spock-tts/models/omnivoice \
  -reference-tokens /workspace/tmp/omnivoice-reference-tokens.json \
  -text 'The evidence is insufficient, Captain.' -frames 75 -steps 8 \
  -threads 2 -gemm-workers 2 -resident-mib 2048 -output NEW.wav
```

`BenchmarkHeadSpanSelection` measures the span planner alone: 114–132 ns and
zero allocations. A constructed 210-token/75-target case projects 78 aligned
rows while dense, versus 6 when only the last target time is active. This is
work-count evidence, not a whole-model speedup estimate.

Tests cover sparse gaps and final tails across streaming, worker, resident and
prepacked modes, unchanged inactive output slots, malformed internal shapes,
changed request prefixes, target shrink/restore, exact legacy token/RNG parity,
and zero generation allocations. Affected tests, vet, race/no-CGo checks and
Linux ARM64 cross-build pass. The repository-wide build still fails in the
previously recorded SpacemiT and DiffusionGemma packages.

## Opt-in shared CFG layer traversal (2026-09-14)

`-shared-traversal` runs conditional and unconditional forwards layer by layer,
loading each streamed float32 decoder layer once for both branches. Distinct
branch activations and attention sequences retain their original SIMD row
shapes. This avoids concatenating branches or introducing cross-branch attention.
The mode requires guided streamed siblings and rejects resident/prepacked and
direct-Q8 modes. It is available in `generate`, `synthesize`, `synthesize-long`
and `serve`; JSON output/worker startup reports `shared_traversal`.

For the 28-layer model and an eight-positive-step schedule, decoder layer loads
fall from 448 to 224. Embeddings, heads and the transformer arithmetic are still
computed separately. Resident mode already avoids repeated decoder loading, so
shared traversal offers no loading reduction there.

Matched streamed runs used the preceding section's 75-frame, eight-step prompt,
with two threads/two GEMM workers and **without** `-resident-mib`. Both binaries
were the same build: the after runs added `-shared-traversal`.

| Trial | Separate wall seconds | Shared wall seconds |
| --- | ---: | ---: |
| First | 60.841 | 56.041 |
| Repeat | 60.585 | 61.433 |
| Mean | 60.713 | 58.737 |

The mean is 3.3% lower, but the second trial regressed. The mode stays opt-in;
these two pairs do not establish a consistent speedup. Peak RSS is about
1,114,000 KiB in both modes (1.06 GiB), with no resident decoder cache. All four
WAV hashes equal the baseline hash in the preceding section. Local records are
`/workspace/tmp/omnivoice-paired-{before,after}*.json`.

Exact per-branch logit, token and RNG parity tests cover different branch
lengths, sparse head spans, serial/worker GEMM, target shrink/restore, and
zero-allocation generation. Cancellation tests cover pre-call and mid-traversal
failure followed by successful reuse; execution context references are cleared.
Unsupported combinations fail validation. Tests, vet, race/no-CGo checks and
Linux ARM64 cross-build pass. Repository-wide build failures remain in the
previously recorded unrelated packages.

## Guidance scale experiment (2026-09-14)

The same 75-frame/eight-step short English prompt was run with streamed weights,
two threads and two GEMM workers. Each scale was measured once. Shared traversal
was off. The command in the preceding sections applies with `-guidance SCALE`
and without `-resident-mib`.

| Scale | Wall seconds | Peak RSS KiB | PCM16 RMS | PCM16 peak | Clipped samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 (default) | 60.264 | 1,114,432 | 1,080.48 | 15,953 | 0 |
| 1 | 62.794 | 1,114,116 | 906.62 | 13,077 | 0 |
| 0 | 37.126 | 1,109,324 | 476.55 | 7,866 | 0 |

All three 3-second WAVs transcribed to “The evidence is insufficient, Captain.”
using the existing Azure en-US recognition check. Scale 2 remains byte-identical
to the established WAV hash. Scale 0 is 38.4% faster in this one comparison and
skips one of the two branch forwards; the conditional sequence is longer, so
skipping the unconditional branch does not halve total runtime. Nonzero scale 1
still runs both branches and offers no compute saving. These are quality-changing
settings; the default stays 2. Short ASR success does not qualify voice similarity,
prosody, long/multilingual output or accent. Listening acceptance is open.

Local evidence: `/workspace/tmp/omnivoice-guidance{0,1,2}{,-metrics}.json`,
`/workspace/tmp/omnivoice-guidance-validation.json`, and
`/workspace/tmp/synthetic-spock-guidance{0,1,2}.wav`.
Tests cover five scales, exact same-scale legacy/RNG parity, zero generation
allocations, no unconditional scratch access at zero, invalid/overflow/underflow
scales, shared-zero rejection, and worker nil-default versus explicit zero.
Affected tests/vet/race/no-CGo checks and the Linux ARM64 build pass.

## Column-parallel GEMM workers (2026-09-14)

The new opt-in `-gemm-columns` partitions full 16-column weight panels among
persistent workers, instead of splitting activation rows and repacking the same
weights in each worker. It requires positive `-gemm-workers`. Full row/tail
arithmetic is preserved; leftover columns use the existing fallback. The library
entry points are `GEMMPool.RunColumns` and `Backbone.EnableColumnWorkers`.

A fresh streamed, two-worker/eight-step profile attributed 59% of CPU samples to
the GEMM microkernel, 11% to F16 conversion and 10% to packing. The profiled run
is excluded from timing comparisons. Unprofiled 75-frame short-English runs:

| Trial | Row workers (s) | Column workers (s) |
| --- | ---: | ---: |
| First | 58.753 | 51.706 |
| Repeat | 60.751 | 57.149 |
| Mean | 59.752 | 54.428 |

The column mean is 8.9% lower; both pairs improved. Four WAVs are byte-identical
to the established baseline. Peak RSS remains about 1.06 GiB. Local records:
`/workspace/tmp/omnivoice-columns-{before,after}*-metrics.json`. Commands use
the earlier streamed eight-step recipe with two GEMM workers, adding
`-gemm-columns` only for column runs. Defaults are unchanged.

One additional run combining columns with `-shared-traversal` took 48.382 s;
this combination still needs repeated measurements. The column-only change is
independently validated. Tests cover exact strided/tail/prepacked kernel results,
alpha, cancellation/draining, lifecycle, zero allocations, and model token/RNG/
logit parity across streamed, resident, prepacked and shared-CFG modes. Race,
vet, no-CGo checks and Linux ARM64 cross-build pass.

## Combined worker/cache measurements (2026-09-14, 1ec0bc70)

Resident float32 weights plus column workers had the lowest mean in this batch.
The ten unprofiled runs use the same three-second prompt, 75 frames, eight steps,
guidance 2, two threads and two workers. Setup and WAV output are included.

| Mode | Trial 1 (s) | Trial 2 (s) | Mean (s) | Maximum RSS (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Streamed, row workers | 57.320 | 65.271 | 61.296 | 1.063 |
| Streamed, column workers | 97.373 | 56.560 | 76.966 | 1.100 |
| Streamed, columns + shared CFG | 60.086 | 54.855 | 57.470 | 1.063 |
| Resident, column workers | 52.864 | 50.294 | 51.579 | 2.659 |
| Resident + prepacked, column workers | 58.163 | 55.052 | 56.608 | 4.289 |

All ten WAVs have the established SHA-256
`e2c01684385c2b561d4b086f1ba23fdfb7c9cf64dc227f32df91edb7f665d579`.
[Machine-readable results](combined-workers-2026-09-14.json) retain trial order,
per-run wall/CPU time, RSS, cache setup time, hashes and available `/proc/stat`
snapshots. No outlier is removed. The columns-only first trial took 97.37 s;
its cause was not established. The snapshots were added after that trial and
cannot diagnose its slowdown retrospectively. These results limit the earlier
columns-only speedup claim to its original two pairs.

Resident+columns reduced the mean by 15.9% relative to streamed row workers in
this batch, at about 2.5 times the peak memory. Shared CFG+columns reduced it by
6.2% without the resident-memory cost. Two trials per mode and visible timing
variation are insufficient to establish universal rankings or change defaults.
Prepacking added 3.99–10.54 s setup and did not improve total command time over
raw resident weights; persistent-worker amortisation needs separate testing.

For this VM, try these explicitly selected configurations:

- Lower-memory candidate: `-gemm-workers 2 -gemm-columns -shared-traversal`.
- Faster measured candidate with RAM available:
  `-gemm-workers 2 -gemm-columns -resident-mib 2048`.

Neither changes guidance, steps or generated audio in these tests. Shared CFG
and resident caching are mutually exclusive. The benchmark validates short
single-shot synthesis; it does not establish long-request or warm-worker speed.

## Uncached persistent-worker throughput (2026-09-14)

Raw resident weights were fastest after startup was excluded. Each mode used
one process and two sequential identical requests. Every request explicitly
requested 75 frames, eight steps, guidance 2 and two column workers. Phrase
cache budget was zero; all six chunk events reported `cache_hit: false` and all
completion events reported zero cache hits/payload. No waveform-cache speedup
is included.

| Mode (all column workers) | Startup (s) | Request 1 (s) | Request 2 (s) | Request mean (s) | Sampled peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared-streamed | 0.488 | 47.885 | 57.677 | 52.781 | 1.090 |
| Raw resident | 1.777 | 48.530 | 48.878 | 48.704 | 2.675 |
| Resident + prepacked | 4.859 | 49.931 | 50.494 | 50.212 | 4.317 |

Prepacking did not beat raw resident weights even with setup excluded. Raw
resident requests averaged 7.7% below shared-streamed and 3.0% below prepacked
in this small batch. Mode order was shared, resident, prepacked; this is one
process per mode, not independent process repetitions. The second shared run
was slower, so rankings need broader workload/repetition evidence.

All six worker WAVs share SHA-256
`a8c66c6dc34c4af88e7f2d34fe17e61dfe81291357046b9582e42aac4ee7d238`.
This differs from the single-shot hash because the worker applies boundary
processing; parity is established across the worker modes and repeats.
[Measured results](worker-throughput-2026-09-14.json) include per-request denoise
and decode times, startup and sampled `/proc/PID/status` high-water RSS. RSS is
sampled, not an exact wait4 measurement. Raw events and private audio remain in
`/workspace/tmp/omnivoice-worker-combined-20260914-v3/`.

Reproduce with a new output directory:

```sh
bun scripts/omnivoice-worker-benchmark.ts bin/omnivoice \
  /workspace/projects/spock-tts/models/omnivoice \
  /workspace/tmp/omnivoice-reference-tokens.json NEW_OUTPUT_DIRECTORY 2
bun test scripts/omnivoice-worker-benchmark.test.ts
```

The driver validates uncached completions and waveform hashes, logs events
incrementally, flushes each stdin request, and separates startup from requests.
A fake-worker test covers the `event` field, explicit zero cache, two sequential
requests, and nested `stages` extraction. Two earlier driver attempts stalled
before submitting any request due to a protocol-field mismatch; neither
produced audio or contributed timings. Final stage fields were extracted from
the retained real event logs after correcting the report field names.

## Grouped-query attention packing (2026-09-14)

Attention now packs K/V once per KV group, retaining contiguous buffers while
query heads in that group execute. Q is still packed per query head. This reuses
buffers only within the current attention operation; no cross-step KV cache or
stale activations are involved. With 16 query heads and 8 KV heads, K/V copying
is halved. Arithmetic, memory reservation and SIMD dispatch stay unchanged.

A direct-strided Q/K/V candidate passed exact output checks but was slower:
about 2.72 versus 2.61 ms at 75 tokens, and 20.99 versus 19.24 ms at 210 tokens.
It remains test-only for reproducible benchmarking. Group-local packed K/V
reuse measured about 2.52 and 18.95 ms respectively (three benchmark runs each,
zero allocations), a small isolated improvement.

Matched full resident+column-worker synthesis results (75 frames/eight steps,
guidance 2, two threads/two workers):

| Trial | Before (s) | KV-group reuse (s) |
| --- | ---: | ---: |
| First | 48.022 | 49.765 |
| Reverse-order repeat | 50.138 | 47.557 |
| Mean | 49.080 | 48.661 |

The 0.9% mean difference with opposite pair directions does not demonstrate
an end-to-end speedup. All four WAVs have the established baseline hash; peak
RSS remains about 2.66 GiB. The production change removes duplicate copy work
and preserves contiguous access. Defaults and quality parameters are unchanged.

`TestAttentionStridedExact` checks both candidates against the copied-head
reference, grouped and ungrouped attention, tails, masks and single-token cases.
Existing generation/logit tests, allocation tests, race/vet/no-CGo checks and
Linux ARM64 build pass. Repository-wide build still fails in the previously
recorded unrelated SpacemiT/DiffusionGemma packages.

Reproduce isolated comparisons with:

```sh
go test ./models/omnivoice -run '^$' \
  -bench 'BenchmarkAttention(KVReuse|Strided)' -benchmem -count=3
```

Local records: `/workspace/tmp/omnivoice-attention-kvreuse-bench.log` and
`/workspace/tmp/omnivoice-kvreuse-{before,after}*-metrics.json`.

## AVX2 microkernel loop unrolling (2026-09-14)

A resident+column-worker CPU profile attributed 73.2% of sampled CPU time to the
6x16 GEMM microkernel, 6.7% to packing and 1.6% to F16 conversion. The measured
profile used the same 75-frame/eight-step/guidance-2 prompt. This differs from
the earlier streamed profile, where conversion accounted for about 11%.

The amd64 microkernel now processes two reduction elements per loop iteration,
with a one-element tail. It preserves each output accumulator's FMA order and
the alpha/C epilogue. ARM64 and portable kernels are unchanged. This reduces
loop/pointer-update overhead without changing scratch or worker dispatch.

Two rounds of three samples per shape compared the original and unrolled kernel
with two column workers. The n=3072 shapes improved roughly 4–5% in both rounds.
The n=1024 shapes had mixed results, including regressions. All trials are in
[the raw benchmark evidence](gemm-unroll-2026-09-14.json); these measurements do
not establish a speedup for every model shape or CPU.

| Full synthesis pair | Original (s) | Unrolled (s) |
| --- | ---: | ---: |
| First (candidate first) | 100.891 | 64.111 |
| Repeat (original first) | 58.976 | 53.916 |

Both pairs favoured the candidate, but the wide baseline spread prevents a
reliable end-to-end speedup estimate. No run is discarded. All four WAVs retain
the established baseline hash. Memory was roughly unchanged outside the noisy
first baseline run. This kernel is shared by other models; broad model/runtime
speedups have not been measured.

Same-host old/new fingerprints match over K=1..3072 selected odd/even/boundary
lengths, M/N tails, padded strides, nonzero C and negative nonunit alpha.
`TestGEBPOrderFingerprint` prints the deterministic fingerprint for revision
comparison; it is not hardcoded across architectures with different reduction
orders. Existing SIMD/model tests, race/vet/no-CGo checks and ARM64 build pass.
A focused assembly review verified loop bounds, offsets, FMA order and ABI.
The kernel's existing caller contract requires positive K.

## Rejected four-way microkernel unrolling (2026-09-14)

Four-way K unrolling did not provide a repeatable gain over the production
two-way AVX2 kernel. The production assembly was restored unchanged. Each round
used three samples per shape, two column workers, streamed packing and K=1024;
round two reversed the implementation order.

| Shape M×N | Round 1: 2x / 4x median (ms) | Round 2: 2x / 4x median (ms) |
| --- | ---: | ---: |
| 75×1024 | 2.045 / 2.138 | 2.075 / 2.038 |
| 75×3072 | 6.210 / 6.248 | 6.331 / 6.580 |
| 210×1024 | 5.019 / 5.027 | 5.313 / 5.153 |
| 210×3072 | 15.525 / 16.273 | 15.539 / 15.250 |

The candidate preserves the two/single-element tails and per-output FMA order.
Kernel tests passed and its same-host fingerprint matched the production value.
No full synthesis run was warranted after the inconsistent kernel results.
[Raw logs](experiments/gebp-unroll4-2026-09-14.json) and the
[rejected patch](experiments/gebp-unroll4-rejected.patch) preserve the experiment;
the patch is not applied by the build. This result is specific to the N100 VM
and these matrix shapes. Defaults, binaries and generated audio are unchanged
by this documentation-only checkpoint.

## Audio-head dispatch and chunk sizing (2026-09-14)

Retain column workers for small audio-head projections. At N=128, K=1024 and
2 execution threads, the median of three 200ms benchmark samples was:

| Rows | Serial (µs) | Row workers (µs) | Column workers (µs) |
| --- | ---: | ---: | ---: |
| 6 | 65.47 | 63.93 | 47.99 |
| 12 | 99.66 | 85.42 | 69.84 |
| 24 | 168.23 | 114.84 | 99.10 |
| 48 | 312.05 | 184.55 | 164.95 |
| 78 | 482.06 | 294.60 | 257.08 |
| 126 | 766.70 | 426.45 | 395.70 |

No serial threshold was added: it would slow every tested shape. Allocation
counts were zero. Raw logs retain occasional benchmark harness byte accounting.

A separate real-checkpoint projection test compared chunk sizes of 128, 256,
512 and 1024 columns, with 85 sequence positions and 75 target frames. Median
projection times were 24.48, 24.56, 24.36 and 25.24 ms respectively. Larger chunks
did not provide a meaningful benefit and need larger head/output buffers.
Production remains at 128 columns. Exact logits match across sizes with dense
and sparse target selections and both row/column pools. Repeated projection
allocation assertions pass. No model or default was changed in this experiment.

[Raw evidence](experiments/head-dispatch-2026-09-14.json) includes all samples.
Reproduce with:

```sh
go test ./backends/simd/runtime -run '^$' \
  -bench BenchmarkGEMMAudioHeadDispatch -benchtime=200ms -count=3
GO_PHERENCE_REAL_OMNIVOICE=/path/to/model go test ./models/omnivoice \
  -run TestRealHeadChunkParity -count=1
GO_PHERENCE_REAL_OMNIVOICE=/path/to/model go test ./models/omnivoice \
  -run '^$' -bench BenchmarkRealHeadChunk -benchtime=200ms -count=3
```

The chunk-resizing helper is test-only. These experiments do not demonstrate an
end-to-end improvement; no extra synthesis runs were needed to reject them.

## Masked-row sampler computation (2026-09-14)

Generation skips guided log-probability normalisation and token prediction for
already-revealed flattened codebook/time rows. Public dense sampler APIs remain
available. Noise arrays are still filled and validated at their original dense
sizes, preserving seeded RNG state. Revealed output slots are left untouched.

Confidence top-k now excludes revealed rows explicitly, rather than only using
an `-Inf` sentinel. Review found that the sentinel could tie with a masked row's
`-Inf` score and allow a lower-index revealed row to be selected. A regression
test verifies revealed tokens remain unchanged in this case. This is a bug fix
for exceptional scores; normal finite-score selection order stays unchanged.

For 8 codebooks × 75 frames × 1025 vocabulary entries, guidance 2 and greedy
class prediction, isolated sampler times fall with the masked fraction:

| Rows still masked | Dense time (ms, approx.) | Masked-only time (ms, approx.) |
| --- | ---: | ---: |
| All | 12.7 | 12.3 |
| Half | 12.5 | 6.2 |
| Quarter | 12.1 | 3.0 |
| Eighth | 12.2 | 1.5 |

All runs report zero allocations. This is a small component of total inference.
One full resident+column-worker pair took **45.437 s before / 48.070 s after**;
no end-to-end speedup is demonstrated. Both WAVs match the baseline hash and
peak memory is unchanged. [Raw evidence](experiments/masked-sampler-2026-09-14.json)
retains every sampler sample and both full-run measurements.

Exact masked-row logits/predictions/confidence, untouched revealed slots,
guidance on/off, stochastic/greedy classes, malformed shapes and allocation
checks pass. Existing legacy generation tests still pass exact token/RNG parity.
Race/vet/no-CGo checks and Linux ARM64 build pass. Defaults are unchanged.

## Top-k final ordering (2026-09-14)

Large top-k selections now sort the existing worst-first heap in place instead
of running a quadratic insertion pass. Selections of at most 32 entries retain
insertion sort. A selected NaN also retains the old ordering path, because the
existing score comparator is not a total order for NaN. Selection itself,
score/index tie-breaking, RNG consumption and buffer sizes are unchanged.

Median isolated top-k times (three samples, including selection and ordering):

| Selected k | Insertion final pass (µs) | Heap final pass (µs) |
| --- | ---: | ---: |
| 16 | 3.75 | 3.81 (same insertion implementation) |
| 32 | 5.85 | 5.70 (same insertion implementation) |
| 64 | 10.19 | 8.59 |
| 103 | 17.74 | 12.82 |
| 300 | 77.05 | 39.07 |
| 600 | 241.57 | 67.92 |
| 2000 | 2397.91 | 200.11 |

These tests use max(1025,k) input scores and zero allocations. The larger cases
benefit most: about 3.6x at k=600 and 12x at k=2000. Greedy short synthesis spends
little time here, so these ratios do not represent whole-model gains. A real
75-frame/eight-step/resident-column verification run took 43.625 s and retained
the baseline WAV hash; no matched end-to-end speedup was measured.

Exact old/new ordering tests cover cutoff boundaries, random scores, finite
ties, infinities, NaNs, masked rows and 1000 mixed exceptional-value trials
including signed zero. Existing seeded generation/token/RNG parity tests pass,
as do race/vet/no-CGo checks and Linux ARM64 build. A focused review found no
ordering blocker. [Raw evidence](experiments/topk-sort-2026-09-14.json) preserves
all samples. Reproduce with `go test ./models/omnivoice -run '^$' -bench
BenchmarkTopKFinalSort -benchtime=200ms -count=3`.

## Matched prepared-input upstream comparison (2026-09-14)

Go was faster in both matched-input pairs. Both runtimes consumed the same
saved prompt IDs (210 conditional / 75 unconditional positions), eight
codebooks, 75 target frames, eight steps, guidance 2 and seed 42 on two N100
vCPUs. Python used torch 2.11.0 CPU float32 with SDPA and two intra-op threads;
Go used two column workers and raw resident decoder weights.

| Recorded stage | Python trial 1 / 2 (s) | Go trial 1 / 2 (s) |
| --- | ---: | ---: |
| Denoising | 65.414 / 67.294 | Not separately isolated |
| Generation including resident setup | — | 47.408 / 40.898 |
| Resident-cache setup (inside Go generation) | — | 3.449 / 0.967 |
| Codec decode only | 1.635 / 1.391 | Not separately isolated |
| Codec load/decode + gain/WAV output | — | 3.130 / 3.084 |
| Recorded total | 74.466 / 72.711 | 50.538 / 43.983 |
| Peak RSS (GiB) | 4.735 / 4.808 | 2.650 / 2.650 |

Mean recorded totals are 73.589 s versus 47.260 s: **1.56x**, or 35.8% less time.
These are matched inputs, not identical timing scopes: Python's total excludes
module import time, gain restoration and file output; Go's total includes gain
and WAV writing but excludes the earlier checkpoint-map open. Python loads its
full audio tokenizer/encoder with the model; Go loads only decoder assets for
this prepared-input operation. Memory compares those respective runtime paths.
Peak RSS is about 44% lower for resident Go; earlier streamed results use a
smaller memory footprint but are not the timings in this table.

The harness replaces only upstream `_prepare_inference_inputs` with the saved
IDs/mask and verifies that the unconditional input equals the target suffix.
Upstream `_generate_iterative`, including its padded two-branch batch, schedule,
noise, prediction and codec call, runs unchanged. Native Go runs each branch at
its actual length. Neither path performs reference encoding/text tokenisation
inside this comparison. Both produce 72000 finite audio samples. Python token
hashes repeat and Go WAV hashes repeat; PyTorch and Go PCG noise differ, so this
does not establish cross-runtime waveform or listening equivalence.

[Raw reports](upstream-matched-2026-09-14.json) preserve stage boundaries and
trial order. The Python harness requires the upstream environment:

```sh
/path/to/upstream/python scripts/omnivoice-upstream-prepared-benchmark.py \
  --model /path/to/model --prompt prepared.json --output NEW_REPORT.json
bin/omnivoice -mode generate -model /path/to/model -input prepared.json \
  -steps 8 -threads 2 -gemm-workers 2 -gemm-columns -resident-mib 2048 \
  -output NEW_AUDIO.wav
```

An initial harness attempt used the wrong codec output indexing and failed;
no result from that attempt enters this table. Two successful runs used
`.audio_values`, matching the existing codec fixture. The codec is a useful
next profiling target, but decode-only Go timings are needed before attributing
the entire stage difference to codec compute. This replaces the earlier rough
1.9x comparison based on different reference/preparation settings.

## Decoder-only comparison and contiguous convolution packing (2026-09-14)

Matched deterministic codec inputs confirm a native decode gap independent of
loading or WAV output. Both use 8×75 codes, `((book*75+t)*13)%1024`, CPU float32,
two execution threads, one untimed warm-up and three measured decode calls.
Go uses prepared reusable scratch; PyTorch uses its standard decoder allocation
path. Model loading, text/reference preparation and postprocessing are excluded.

| Decoder | Decode seconds (three calls) | Mean (s) |
| --- | --- | ---: |
| Upstream torch 2.11 CPU | 0.894 / 0.952 / 0.904 | 0.917 |
| Go baseline | 2.674 / 2.682 / 2.706 | 2.687 |
| Go contiguous-copy candidate | 2.229 / 2.176 / 2.168 | 2.191 |
| Go baseline repeat | 2.691 / 2.684 / 2.694 | 2.689 |
| Go candidate repeat | 2.192 / 2.180 / 2.170 | 2.180 |

The prepared Go decode calls allocate zero bytes. The Go profile (which also
includes benchmark setup/warm-up) attributed about 62% of CPU samples to NN
GEMM and 23% directly to convolution preparation. For stride-one convolution,
positions within each dilated tap are contiguous. Packing now uses bounded
slice copies per tap, retaining cleared scratch for padding; other strides keep
the scalar path. GEMM and bias addition are unchanged.

This lowers measured native decode time by about **18–19%**, roughly half a
second for three seconds of audio. The full 72000-sample float32 hash is exact
before/after:
`f4895c215a7bbb797c84e85555b73b933dc0a7f7ea3f81ef11ccb7436212729a`.
It is not a cross-runtime hash comparison. PyTorch remains about 2.4x faster at
decode alone after this change. No full-synthesis speedup is measured here.

Tests cover scalar-reference parity across frames 1/7/63/64/65/129, kernels
1/3/7, strides 1/2 and dilations 1/3/9, plus real-codec PyTorch fixture parity.
Model/CLI tests, race/vet/no-CGo checks and Linux ARM64 build pass.
[Raw evidence](experiments/codec-copy-2026-09-14.json) retains all runs.

```sh
GO_PHERENCE_REAL_OMNIVOICE=/path/to/model go test ./models/omnivoice \
  -run '^$' -bench BenchmarkRealCodecDecode75Frames -benchtime=1x -count=3
/path/to/upstream/python scripts/omnivoice-upstream-codec-benchmark.py \
  --model /path/to/model/audio_tokenizer --output NEW_REPORT.json
```

## Packed codec GEMM (2026-09-14)

Prepared decoder convolutions now use the existing six-row packed microkernel
for full amd64 tiles. `SgemmNNPackedOverwriteTo` packs NN input panels into
caller-owned scratch, overwrites the output and retains the old NN kernel for
row/column tails, small shapes and other architectures. Build tags select the
architecture path; the existing SIMD availability check still applies.

The wrapper validates shapes and scratch capacity before mutation. Inputs,
output and scratch must satisfy its documented non-overlap contract. The codec
reserves one extra 224 KiB panel in `Prepare`; `DecodeInto` still allocates zero
bytes. The alpha=1 path preserves the measured finite float32 accumulation
results. Tests check NaN classification rather than NaN payload identity.

| Decode-only trial set | Seconds | Mean (s) |
| --- | --- | ---: |
| Initial packed candidate | 1.632 / 1.620 / 1.617 | 1.623 |
| Repeated committed NN path | 2.247 / 2.173 / 2.186 | 2.202 |
| Repeated packed candidate | 1.629 / 1.630 / 1.679 | 1.646 |

The repeat reduces decode time by **25.2%**, about 0.56 seconds for three seconds
of audio. Codes, preparation and warm-up follow the preceding decoder benchmark.
The baseline binary uses the committed codec implementation with the candidate's
unused panel allocation outside timing. All six repeated waveforms match the
established 72000-sample float32 hash. The earlier upstream decode-only mean was
0.917 s; it was not rerun in this experiment.

A full synthesis verification also preserves the established WAV hash. It took
59.269 s, including 16.559 s resident-cache setup and 2.055 s codec load/decode/
save. No matched full-synthesis speedup was measured. Disk space after the broad
build checks was about 223 MiB; the full synthesis result is retained in the raw
record rather than discarded.

Validation passed:

- `make test-omnivoice vet-omnivoice build-omnivoice`.
- SIMD/model/CLI race tests, SIMD vet, no-CGo tests and Linux ARM64 build.
- Real-codec upstream fixture tolerance and zero-allocation checks.
- Exact NN parity across row/column/kernel tails and padded strides, untouched
  output padding, invalid shape/overflow rejection before mutation, signed zero,
  infinities, subnormals and NaN classification.

`go test ./backends/...` failed outside SIMD runtime, including Vulkan wrapper
expectations. `go build ./...` failed in existing SpacemiT/DiffusionGemma code
and later hit insufficient disk space. Neither broad check passed. The delegated
review timed out; the wrapper was reviewed locally and the identified platform
dispatch issue was corrected before testing.

[Raw benchmark and validation logs](experiments/codec-packed-2026-09-14.json)
include all trials and the full synthesis record. Reproduce the GEMM shapes with:

```sh
go test ./backends/simd/runtime -run '^$' -bench BenchmarkNNCodecPacked \
  -benchtime=150ms -count=2
```

## Direct pointwise codec input rejected (2026-09-14)

Removing im2col and result-tile copies from prepared pointwise convolutions did
not improve decode time. The candidate passed native channel-major input and
output directly to the packed NN wrapper for kernel=1, stride=1, padding=0,
then applied bias in the original order. Other convolutions were unchanged.

| Trial set | Tiled baseline mean (s) | Direct candidate mean (s) |
| --- | ---: | ---: |
| Baseline then candidate | 1.723 | 1.779 |
| Candidate then baseline | 1.624 | 1.632 |

Each set contains three decode-only samples with an untimed warm-up per sample,
using the existing deterministic 8-codebook/75-frame benchmark. All twelve
waveform hashes match and all timed calls allocate zero bytes. The first set
is slower with direct input; the reverse-order repeat is effectively tied.
The cause was not isolated. Production retains the bounded tiled path from
`076723e0`.

The [candidate patch](experiments/codec-pointwise-rejected.patch) is unapplied;
[raw measurements](experiments/codec-pointwise-2026-09-14.json) retain every
sample. After rejection, `codec.go` was verified byte-for-byte against HEAD and
`make test-omnivoice vet-omnivoice` passed. Test binaries used `/tmp` tmpfs due
to the low workspace disk space. No full-synthesis timing was run for this
rejected candidate.

## Packed decoder profile and tile-size sweep (2026-09-14)

The current packed decoder profile attributes about 51% of CPU samples to the
GEBP microkernel, 6% to the NN kernel, 8% to memory copies and 4% to clearing.
Snake activation accounts for about 12% cumulatively. The profile includes
model loading and warm-up; its timing is excluded from the comparisons below.

Larger convolution tiles did not establish enough benefit to increase scratch:

| Tile positions | First round mean (s) | Reverse round mean (s) | Extra scratch |
| --- | ---: | ---: | ---: |
| 64 (production) | 1.629 | 1.679 | — |
| 128 | 1.614 | 1.615 | 1.125 MiB |
| 256 | 1.646 | 1.640 | 3.375 MiB |

Each round contains three samples. The reverse baseline includes a 1.787 s
sample, retained in the evidence. Three subsequent interleaved 64/128 pairs
average 1.627/1.613 s (0.9% difference); one pair favours the baseline. Keep 64
positions: the measured 128 gain is small and mixed, and 256 offers no gain.

Only the convolution tile constant and its packed/result scratch allocations
changed in the candidates. All 24 measured decode calls allocate zero bytes and
retain the established float32 waveform hash. Production files match HEAD
exactly and `make test-omnivoice vet-omnivoice` passes after the experiment.
No full-synthesis timing was run. Test binaries used `/tmp` because the workspace
disk is nearly full.

[Profile, raw trials and exact candidate recipe](experiments/codec-tiles-2026-09-14.json)
preserve the experiment, including the outlier and run order.

## Snake post-sine arithmetic (2026-09-14)

On amd64, Snake now squares, scales and adds each sine result in one assembly
loop. Separate multiply/multiply/add instructions preserve the float32 rounding
boundaries; no fused multiply-add is used. The sine algorithm, 256-element
scratch chunks and SIMD/scalar sine dispatch are unchanged. Other platforms and
amd64 without `HasVecAsm` retain the three existing vector passes. The capability
check requires AVX2+FMA on amd64.

| Implementation | First Snake mean (µs) | Repeat Snake mean (µs) |
| --- | ---: | ---: |
| Existing vector passes | 15.914 | 15.964 |
| Combined scalar Go loop (rejected) | 18.438 | — |
| Combined amd64 assembly loop | 15.307 | 15.688 |

Each cell averages three samples of the existing 4096-element benchmark,
including its input copy. Repeat medians are 15.890 µs baseline and 15.396 µs
assembly. All samples allocate zero bytes. Real-codec decode means are
1.607/1.595 s baseline/assembly in the first round and 1.625/1.599 s in the
reverse-order repeat. The measured benefit is small (about 0.8–1.6% for decode).
No matched full-synthesis speedup was measured.

Bitwise tests compare the old and new channel paths across vector tails,
256-element boundaries, long channels and several alpha values. Separate tests
cover sentinel bounds, signed zero, subnormals, infinity and NaN classification.
Real-codec float32 hashes and the established full-synthesis WAV hash match.
Scratch usage and decode allocations are unchanged. The private assembly helper
requires equal-length, non-overlapping slices, supplied by `snakeChannel`.

`make test-omnivoice vet-omnivoice`, model/CLI race and no-CGo tests, real-codec
fixture tests, native build and ARM64 cross-build pass. A focused independent
review found no ABI, bounds, tail or rounding blocker. `go build ./...` still
fails outside the changed code, including SpacemiT and DiffusionGemma packages;
the full log is retained. Build temporaries used `/tmp` due to low workspace
space (169 MiB after validation).

[Raw samples and validation evidence](experiments/snake-post-2026-09-14.json)
and the [rejected scalar-Go patch](experiments/snake-go-fusion-rejected.patch)
retain both candidates. Reproduce the microbenchmark with:

```sh
go test ./models/omnivoice -run '^$' -bench 'BenchmarkSnakeChannel/simd' \
  -benchtime=300ms -count=3
```

## Bounded transposed-convolution overlap-add (2026-09-14)

Transposed convolution now computes the valid kernel-tap range once per input
position. Each channel accumulates bounded slices instead of checking output
bounds for every tap. The input-position/channel/tap iteration order is unchanged,
so each output receives the same float32 additions in the same order. Scratch
capacity and allocations are unchanged.

| Decode-only round | Baseline mean (s) | Bounded-loop mean (s) |
| --- | ---: | ---: |
| Baseline then candidate | 1.592 | 1.563 |
| Candidate then baseline | 1.595 | 1.584 |

Each mean contains three samples of the existing deterministic 8×75 benchmark,
with an untimed warm-up per measured invocation. The gain is small, about
0.7–1.8%. All twelve codec waveform hashes match, with zero timed allocations.
One full synthesis verification also matches the established WAV hash; no matched
full-synthesis speedup was measured.

Exact reference tests span frame counts 1/2/7/31/32/33/65, kernels 1/3/7/16,
strides 1/2/3/8, padding 0/1/4/16 and output padding 0 or stride-1, restricting
the matrix to positive output lengths. This includes completely clipped input
positions and both sides of the 32-frame tile boundary. Independent review found
no bounds or accumulation-order blocker for valid parameters.

`make test-omnivoice vet-omnivoice`, model/CLI race and no-CGo tests, the real-codec
fixture, native build and ARM64 build pass. `go build ./...` still fails outside
the changed code, including SpacemiT and DiffusionGemma. The experiment used
`/tmp` for build temporaries; workspace space was 132 MiB after validation.
[Raw samples, synthesis record and build log](experiments/codec-scatter-2026-09-14.json)
retain all measurements and failures.

## Remove duplicate GEMM output clears (2026-09-14)

Prepared codec GEMM already clears its output in `SgemmNNPackedOverwriteTo`.
The convolution and transpose callers no longer clear the same output first.
The unprepared accumulating NN path clears its output inside the codec GEMM
helper, preserving overwrite semantics in both modes. Im2col padding clears
and overlap-add output initialisation are unchanged.

| Decode-only round | Baseline mean (s) | Single-clear mean (s) |
| --- | ---: | ---: |
| Baseline then candidate | 1.586 | 1.568 |
| Candidate then baseline | 1.583 | 1.564 |

Each mean contains three samples with one untimed warm-up per measured call,
using the existing deterministic 8×75 benchmark. The measured saving is about
1.1–1.2%. All twelve float32 codec hashes match; timed allocations remain zero.
One full-synthesis verification retains the established WAV hash. No matched
full-synthesis speedup was measured. Scratch capacity is unchanged.

New tests initialise reused GEMM output with NaNs and check exact results,
tail sentinels and zero allocations for prepared and unprepared paths, including
full packed tiles, small fallbacks and row/column tails. Independent review
found no overwrite-contract blocker. `make test-omnivoice vet-omnivoice`, race,
no-CGo, real-codec fixture, native build and ARM64 build checks pass.
`go build ./...` still fails in unrelated packages. Build temporaries used `/tmp`;
workspace free space fell to 94 MiB, so further build-heavy work needs disk space.

[Raw trials, synthesis record and build failure log](experiments/codec-clear-2026-09-14.json)
retain all evidence.

## Cumulative codec qualification (2026-09-14)

A fresh matched comparison of `21fcb4bb` (before codec changes) and `3c5a7cb6`
confirms the cumulative codec saving while preserving exact output.

| Measurement | Before mean (s) | Current mean (s) |
| --- | ---: | ---: |
| Decode only, three interleaved runs | 2.945 | 1.618 |
| Codec load/decode/save, two full runs | 3.229 | 2.046 |
| Full synthesis, two runs | 46.824 | 46.695 |

Decode-only time is about 45.0% lower (1.82x faster). Codec load/decode/save is
about 36.6% lower. Full-synthesis pair directions differ, and the 0.3% mean
change does not establish a meaningful whole-run gain. Generation means vary
from 43.595 s in the baseline to 44.649 s in the current build, masking most
of the codec-stage saving. The transformer implementation/settings are unchanged
between these commits.

The old production source was built in a detached temporary worktree with only
the current `codec_bench_test.go` added as an untracked harness. Decode uses the
same deterministic 8×75 codes and one untimed warm-up. All six decode hashes
match, with zero allocations. All four full-synthesis WAVs match. Full runs use
the same prepared IDs, 75 frames, eight steps, guidance 2, two column workers
and 2048 MiB resident budget. No reference preparation is included.

The fresh baseline decode mean is slower than its earlier 2.69 s measurement;
all new samples are retained as a separate comparison. `make test-omnivoice
vet-omnivoice` passes. This qualification changes no production code.

Before building, `go clean -cache` reclaimed about 3.1 GiB of rebuildable Go
compilation cache, increasing free disk space from 94 MiB to 3.2 GiB. Model
exports, source, benchmark logs and audio were preserved. After rebuilding and
testing, about 3.1 GiB remained free. Temporary binaries/worktree used `/tmp`.

[Raw trials, run order and provenance](experiments/codec-cumulative-2026-09-14.json)
record the matched comparison.

## Align transposed-convolution tiles to the packed kernel (2026-09-14)

Prepared amd64 transposed convolutions now use 30-row tiles when SIMD GEMM is
available. Thirty is divisible by the six-row packed microkernel size; the
previous 32-row tiles sent two rows through the NN fallback on each full tile.
Unprepared decoding and other architectures retain 32-row tiles. Scratch remains
sized for 32 rows, so memory capacity and allocation behaviour are unchanged.

| Decode-only round | 32-row mean (s) | 30-row mean (s) |
| --- | ---: | ---: |
| Baseline then candidate | 1.583 | 1.548 |
| Candidate then baseline | 1.566 | 1.541 |

Each cell contains three samples with an untimed warm-up, using the existing
8×75 deterministic codec benchmark. Timings fall about 1.7–2.2%; all twelve
codec hashes match and timed allocations remain zero. The initial candidate
used 30 rows unconditionally; the final platform/preparation gate selects that
same measured path on this amd64 host.

Changing tile boundaries preserves increasing global input-row order and hence
the overlap-add sequence. Exact tests compare prepared output with the old
32-row reference around both boundaries, including frames 29/30/31/32/33 and
59/60/61. A focused independent review found no blocker. Final-code real-codec
fixture and full-synthesis WAV parity pass, as do affected make tests/vet,
race/no-CGo tests, native build and ARM64 build. `go build ./...` fails in
unrelated packages. No matched full-synthesis speedup was measured.

[Raw trials, synthesis verification and build failures](experiments/codec-transpose-aligned-2026-09-14.json)
retain the evidence and candidate provenance.

## Codec length qualification (2026-09-14)

The cumulative codec gains hold at all six tested lengths, including the public
250-frame maximum. Current `4d7adca1` and original `21fcb4bb` produce identical
float32 hashes at each length, with zero allocations during every timed call.

| Frames | Audio (s) | Original decode (s) | Current decode (s) | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.04 | 0.053 | 0.043 | 20.0% |
| 2 | 0.08 | 0.101 | 0.054 | 46.3% |
| 30 | 1.20 | 1.116 | 0.614 | 45.0% |
| 32 | 1.28 | 1.148 | 0.640 | 44.2% |
| 75 | 3.00 | 2.751 | 1.594 | 42.1% |
| 250 | 10.00 | 9.114 | 5.209 | 42.8% |

Each cell averages two samples with one untimed warm-up per benchmark invocation;
version order was reversed for the second round. The same current test harness
was copied into a detached original worktree without production changes. Input
codes follow `((book*frames+t)*13)%1024` for eight codebooks. Different frame
counts have different code sequences; parity is checked between implementations
at each length. All 24 measured calls and their hashes are retained.

`BenchmarkRealCodecDecode75Frames` remains available and now shares a helper
with `BenchmarkRealCodecDecodeLengths`. Model loading, preparation and output
allocation precede timing. This adds benchmark coverage only; production code
is unchanged. `make test-omnivoice vet-omnivoice` passes. Listening and
full-synthesis performance are separate from this synthetic decoder test.

```sh
GO_PHERENCE_REAL_OMNIVOICE=/path/to/model go test ./models/omnivoice \
  -run '^$' -bench BenchmarkRealCodecDecodeLengths -benchtime=1x -count=2
```

[Raw length sweep and verified hashes](experiments/codec-lengths-2026-09-14.json)
include trial order and baseline provenance.

## Skip clears for fully overwritten activations (2026-09-14)

Codec scratch acquisition now distinguishes cleared accumulators from buffers
that callers fully overwrite. Embedding input, residual-copy work and convolution
output use `bufferOverwrite`; RVQ sums and transposed-convolution overlap-add
outputs retain cleared `buffer` acquisition. Padding scratch is still cleared.
The three changed callers initialise every element before reading it.

| Decode-only round | Baseline mean (s) | Overwrite-buffer mean (s) |
| --- | ---: | ---: |
| Baseline then candidate | 1.553 | 1.519 |
| Candidate then baseline | 1.530 | 1.503 |

Three samples per cell use the same deterministic 8×75 codes and one untimed
warm-up. The saving is about 1.8–2.2%; all twelve hashes match with zero timed
allocations. Scratch capacity is unchanged. Unprepared allocations remain
zero-initialised by Go.

The real-codec test now poisons all activation slots and other scratch with NaNs,
checks exact output, decodes at a smaller prepared length, restores the original
length, cancels and retries. Exact hashes at all six length-sweep sizes and the
full-synthesis WAV hash also match. The separate length verification was slower
than earlier runs and is retained without a performance conclusion.

Independent review found no initialisation/lifetime blocker. Affected make
tests/vet, race/no-CGo tests, native and ARM64 builds pass. The real-codec poison
test was executed with local weights. Full-tree build still fails in unrelated
packages. No matched full-synthesis gain was measured.

[Raw paired trials, length verification, synthesis and build log](experiments/codec-activation-2026-09-14.json)
retain all evidence.

## Packed attention probability/value product (2026-09-14)

Attention now reuses query-head scratch for packed probability×value GEMM after
QK and softmax finish. Query scratch is overwritten before the next head uses it.
For head width d>=16, its tokens*d capacity covers the tokens*16 packing panel.
Smaller widths retain the original NN call; the packed wrapper retains platform
and small-shape fallbacks. No additional workspace or allocations are required.

The current resident/column-worker profile attributes about 74% of CPU samples
to the packed microkernel, 7% to packing and 8% cumulatively to attention. This
whole-command profile includes setup and codec; its timing is excluded below.

| Attention tokens | Baseline mean (ms) | Packed AV mean (ms) | Reverse baseline / packed (ms) |
| --- | ---: | ---: | ---: |
| 75 | 2.531 | 2.446 | 2.522 / 2.416 |
| 210 | 18.786 | 17.388 | 19.007 / 17.541 |

Each mean contains three samples. The isolated improvement is about 3–8%, with
zero allocations. Full-synthesis totals are 43.806/46.553 s baseline and
46.661/46.253 s candidate: means 45.180/46.457 s with opposite pair directions.
No whole-run gain is established. All four WAVs match the established hash.

Exact attention tests now cover widths 15/16/17, packed row tails, grouped-query
reuse and NaN-filled scratch, compared against the original packed reference.
Existing generation token/RNG parity, model/CLI tests, race/vet/no-CGo tests,
native and ARM64 builds pass. Independent review verified scratch capacity,
non-overlap and lifetime. Full-tree build still fails in unrelated packages.

[Raw microbenchmarks, full trials, profile and build log](experiments/attention-packed-nn-2026-09-14.json)
retain all measurements, including the slower full-run mean.
