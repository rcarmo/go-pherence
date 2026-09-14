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
