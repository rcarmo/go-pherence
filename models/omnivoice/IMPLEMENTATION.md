# Native inference milestone: backbone, denoising, decode

## Implemented and tested

- Full mixed text/audio forward: each audio position sums the eight codebook
  embeddings, text uses codebook-zero text IDs, then all 28 Qwen3 layers, final
  RMSNorm and chunked audio-head projection. Returns `[codebook,time,vocab]`.
- Fixed-capacity `Backbone.ForwardInto` streams weights using the reusable arena;
  all reads and calculations are native Go. Zero successful-call allocations.
- Sampler: float32 shifted timetable, reveal counts, normalized classifier-free
  guidance, mask-token suppression, top-k, Gumbel perturbation, confidence/layer
  penalty and filled-position masking. Scratch is preallocated.
- `Generation.GenerateInto` connects conditional/unconditional forwards to the
  reveal loop. Go PCG RNG is seeded per invocation, not PyTorch-compatible.
  Position temperature is constant as in upstream, not annealed.
- Decode-only HiggsAudioV2: RVQ embedding/projection sum, fc2, DAC input conv,
  five transposed-convolution upsampling blocks, residual convolutions with
  dilations 1/3/9, Snake activations and output convolution. Final tanh is omitted
  as required by Higgs. SIMD GEMM handles convolutions using bounded im2col tiles;
  scatter and sine remain scalar. `Prepare` reserves reusable activation slots,
  weight lookups and convolution scratch; `DecodeInto` allocates nothing after setup.
  `Decode` remains an allocating convenience wrapper. A decoder is single-caller.
- Validated fixed codec variant: 24 kHz, 8 quantizers, codebook size 1024,
  decoder rates 8/5/4/2/3. Decode limit is 250 frames (10 seconds).
- WAV output exclusively creates PCM16 mono files and attenuates only to avoid
  clipping. Original waveform generation is distinct from file gain scaling.

## Numerical evidence

| Check | Result |
|---|---|
| Tiny full forward versus upstream OmniVoice.forward | every logit, max error 2.39e-7 |
| Real full forward, 3 mixed positions | all 24,600 logits, max error 0.000153, mean 1.34e-5 |
| Real forward allocation counters | 0 bytes, 0 allocations after setup |
| Tiny 4-step greedy generation | matches upstream-derived loop output/schedule |
| Sampler random primitives | upstream helpers with explicit uniform fixtures |
| Tiny complete generation allocations | zero after construction |
| Real decoder, 8 codebooks × 2 frames | all 1,920 samples, max error 5.59e-7 |

The real forward took ~2.34 seconds with streamed weights. The Python figure
includes model/codec loading, so these timings are **not** a valid speedup ratio.
Sampling tie order can differ from Torch: Go uses lowest flattened index.

The top-k ratio uses float64, matching Python scalar semantics. In float32,
`0.1 * 20` followed by double-precision `ceil` can produce 3 instead of 2; a
reference fixture exposed this and the API was corrected before integration.

## CLI

```sh
make test-omnivoice vet-omnivoice build-omnivoice
bin/omnivoice -mode logits -model "$MODEL" -input token-input.json
bin/omnivoice -mode generate -model "$MODEL" -input prepared-prompt.json \
  -steps 8 -output synthetic.wav
```

`logits` input: `tokens`, flattened `[codebook,time]` `ids`, `audio_mask`, optional
`positions` and additive `mask`. CLI limits to 256 positions.

`generate` input: `conditional` and `unconditional` objects with `tokens`, `ids`
and `audio_mask`; plus `target_frames`, descriptive `text` and `reference`.
Target frames occupy the end of each sequence. Maximum input length is 512
positions. It speaks the **supplied tokens**; descriptive text is not verified
against them and is not tokenized by this command. Generate currently uses full
attention/implicit positions, so custom masks/positions are not supported.

## Preparation boundary

`scripts/omnivoice-export-prompt.py` uses the existing Python reference runtime
to tokenize one text and encode one reference WAV. It writes a prepared JSON.
This is a development bridge, not part of the Go executable and not a claim of
fully native arbitrary text/reference input. Go runs all subsequent iterative
inference, codec decoding and WAV writing without Python subprocesses.

`prepare` and `synthesize` now tokenize text natively using cached reference codes:

```sh
bin/omnivoice -mode prepare -model "$MODEL" -reference-tokens voice-codes.json \
  -text 'The evidence is insufficient, Captain.' -frames 75 -output prompt.json
bin/omnivoice -mode synthesize -model "$MODEL" -reference-tokens voice-codes.json \
  -text 'The evidence is insufficient, Captain.' -frames 75 -steps 8 \
  -output synthetic.wav
```

Cache fields: `books`, `frames`, flattened `[book,frame]` `codes`, exact reference
`transcript`, optional original `ref_rms`. Native encoding is available as described below.
Use matching codes/transcript from the same recording. Language defaults to `en`,
denoise to true; `-instruct` is optional. Maximum combined prompt is 512 positions.
Qwen single-digit splitting and cached merge ranks pass a nine-case tokenizer
fixture. The complete real prompt matches the Python export at all 210 positions.
Run that private test with `GO_PHERENCE_REAL_PROMPT` pointing to a JSON object
containing `reference` (cache) and `expected` (Python-exported prepared prompt),
plus `GO_PHERENCE_REAL_OMNIVOICE` pointing to the model directory.

Generation restores reference loudness by multiplying by `ref_rms / 0.1` when
below 0.1, before peak limiting. This matches upstream's RMS restoration.

Keep reference-derived prepared prompts private just like reference recordings.
They are not fixtures/source exports. The checked-in tiny checkpoint is random
synthetic test data, and the decoder fixture uses synthetic code IDs.

## Real native synthesis and packed SIMD

Two 3-second synthetic Nimoy-conditioned samples completed using one prepared
prompt, eight steps and seed 42. Both external ASR checks recovered “The evidence
is insufficient, Captain.” Native baseline took 496.999 s (492.631 s generation,
4.368 s decode/save). Packed SIMD took 149.283 s (143.607 s generation, 5.675 s
decode/save), about 3.3× faster in these runs. This is not a controlled comparison
with Python, whose RNG and postprocessing differ. Peak observed generation RSS
was roughly 1.06 GB, not a formal whole-process maximum measurement.

Added `SgemmNTPackedTo`: checked slices, caller-owned panel scratch, existing
amd64 6×16/arm64 4×16 GEBP microkernel and safe tail fallbacks. This avoids repeated
B dot-product scans across rows. Shape/alpha/zero-allocation tests and race tests
pass. Real 128-token block fell from ~766 ms to ~126 ms with zero allocations;
real-block prefix/aggregate parity remained within float32 tolerances. Tiny full
logit fixtures use a small-row fallback; broader real packed-path parity is
covered by the 128-token block check, not the three-token logits check alone.

Full `go test ./backends/...` was run because a reusable backend entry point was
added. Existing SpacemiT package build/test failures reproduce on untouched
`e4c24e6f`. SIMD runtime tests pass; no architecture assembly was changed.

## Cached-reference profiling and FP16 conversion (2026-09-13)

Same 3-second text, approved chess reference cache, eight steps, seed 42, two
threads. Profiled generation/decode-save times exclude about three seconds of
CLI tokenization/backend discovery/setup. Single runs, not statistical estimates:

| Weight conversion | Generation | Decode/save | Combined |
|---|---:|---:|---:|
| Scalar | 94.75 s | 4.02 s | 98.77 s |
| F16C plus scalar NaN scan | 82.80 s | 4.13 s | 86.93 s |
| F16C plus vector NaN scan | 72.24 s | 4.26 s | 76.50 s |

All three PCM WAVs have SHA-256
`b38d9318dbdf478abdf7054d5da194799a4e5899dde91a6c2cb0661cb4cf3a4b`.
The last CPU profile spans 79.44 seconds including whole-command setup.
GEBP accounts for 58.5% of CPU samples, scalar B packing 9.5%, F16C conversion
6.8%, scalar exponentials 4.2%. Whole-command sampled allocations in the scalar
run were 395 MB, mainly tokenizer JSON, two streamed weight arenas and codec
weights/scratch. This is cumulative allocation, not peak RSS. Backbone forward,
weight reload and prepared codec decode still pass zero-allocation tests.

`simd.F16LittleEndianToF32` uses detected AVX/F16C on amd64 and a scalar fallback.
It validates byte lengths and preserves every scalar half-conversion bit pattern,
including NaN payload/signalling bits via exceptional-block fallback. Tests cover
all 65,536 values, tails, NaN blocks, malformed inputs and allocations. ARM64
cross-build and no-CGo tests pass. Large finite conversion benchmark measured
about 8.3x scalar throughput on this host; small in-cache buffers measured more.

## Native reference encoding (2026-09-13)

`encode-reference` imports mono reference audio through `go-264/audio`, normalises
quiet references to RMS 0.1, trims to the 960-sample hop boundary, resamples to
16 kHz with Hann-windowed sinc, pads 160 samples on both sides, runs HuBERT and
averages its 13 hidden states. It selects every second semantic frame, then runs
DAC acoustic encoding, semantic convolutions, fusion and eight RVQ stages.

```sh
bin/omnivoice -mode encode-reference -model "$MODEL" \
  -reference reference.wav -transcript 'Exact reference transcript.' \
  -output reference-codes.json
bin/omnivoice -mode synthesize -model "$MODEL" \
  -reference reference.wav -transcript 'Exact reference transcript.' \
  -text 'The evidence is insufficient, Captain.' -frames 75 -steps 8 \
  -output synthetic-native.wav
```

The complete raw-reference path runs without Python. It requires 2–20 seconds
and an explicit transcript. Silence removal and ASR are absent. Upstream's
preprocessed chess cache has 104 frames; the raw 4.5-second recording has 112.
Compare matching raw waveforms, not those two different preprocessing paths.
Non-24-kHz source import uses go-264's resampler; exact import parity is verified
for the approved 24-kHz PCM16 recording, not all source codecs/rates.

Verification:
- HuBERT, five output frames: maximum error 4.77e-6 against PyTorch.
- Complete two-second synthetic reference: all 400 codes match exactly;
  resampler maximum error 4.47e-8.
- Complete approved raw chess recording: all 896 codes match exactly.
- Opt-in real tests require `GO_PHERENCE_REAL_CODEC` and
  `GO_PHERENCE_REFERENCE_PYTHON`; default tests never load private models/Python.
- Reference-derived JSON and audio stay outside the repository.
- Native package tests/vet, race/no-CGo checks and ARM64 cross-build pass.
- Nil receivers/context and bounded HuBERT input checks reject invalid calls.
  HuBERT/CodecEncoder instances own scratch and are single-caller.

`Prepare`/`ExtractInto`, `Prepare`/`EncodeFeaturesInto` and the complete
`ReferenceEncoder.Prepare`/`EncodeInto` now pass zero-allocation tests after setup.
Constructors cache immutable operator names/weights; frame-specific capacity
checks are cached. Reference normalisation, sinc resampling and semantic frame
selection reuse staging buffers. Convenience APIs still allocate returned codes
or features. Instances are single-caller; input slices must not alias scratch.
Fixed workspace capacities and loaded float32 weights account for most
whole-command memory.

Full raw-reference synthesis, three seconds, eight steps, seed 42, two threads:
97.91 seconds from CLI entry, including 14.69 seconds reference loading/encoding,
76.27 seconds generation and 4.23 seconds decode/save. The generation JSON's
`command_seconds` includes reference preparation; `total_seconds` retains the
older generation/decode/save boundary. Two native output runs were byte-identical:
`560fe8ae07d6af712f51f64f553d8e8a0b6146778d31a3a944a6db90b61a1e0e`.

Whole-command sampled cumulative allocations fell from 1350.7 to 1317.2 MB after
removing allocate-then-replace scratch calls. The latter includes 704.3 MB codec/
HuBERT/decoder weights, 212.3 MB encoder scratch, 120.1 MB backbone weight arenas,
64.8 MB HuBERT scratch and 37.4 MB decoder scratch. These figures are not peak RSS.
Timing varied between runs; no controlled speedup is attributed to this change.
Latest private profiles: `/workspace/tmp/omnivoice-full-native-v2.{cpu,mem}`.
Output: `/workspace/tmp/synthetic-spock-full-native-v2.wav`. Listening acceptance
and fresh ASR verification of this raw-reference sample have not been performed.

## Prepared encoder optimisation (2026-09-13)

HuBERT linear projections and codec encoder convolutions now use caller-scratch
packed SIMD GEMM. HuBERT attention retains its existing layout. The codec
convolution im2col layout is time-major for packed multiplication; scratch
capacity includes the required `fan * 16` panel. Shape, nil-receiver and input
limits are validated before allocation. CLI preparation validates output suffix,
existing paths, required transcript/text, source selection, frames and steps
before loading models.

A real two-second HuBERT benchmark measured 3.23 s with the old linear path and
1.55 s packed, both 0 B/op and 0 allocs/op. Packed-vs-old max error was 9.78e-6.
Real raw chess reference codes remain exactly 896/896; synthetic full reference
codes remain 400/400. Zero-allocation regression tests cover both compute stages
and the full prepared reference pipeline. Default tests remain opt-in for models
and Python; the real benchmark needs `GO_PHERENCE_REAL_CODEC`.

Latest raw-reference full synthesis: 91.80 s from CLI entry, including 9.91 s
reference loading/encoding, 74.96 s generation and 4.13 s decode/save. Previous
run: 97.91 s, including 14.69 s reference preparation. These are single runs.
The output SHA-256 is unchanged. Profile `/workspace/tmp/omnivoice-full-native-v3`
(`.cpu`, `.mem`, `.json`); output `/workspace/tmp/synthetic-spock-full-native-v3.wav`.
Sampled cumulative allocation remains 1316.5 MB, dominated by model weights and
scratch setup. Prepared zero-allocation execution does not remove those costs.

## Setup-memory and SIMD softmax checkpoint (2026-09-13)

Encoder scratch now has five signal slots rather than eight. Lifetime tests show
four simultaneous slots for exact acoustic alignment and five when padding is
required. The 4.5-second capacity test saves 82,944,000 bytes (79.1 MiB).
`NewBackboneSibling` shares one streamed weight arena between sequential CFG
branches, saving another 60 MiB. Each branch still owns its activation scratch.
Siblings must never run concurrently; repeated generation/zero-allocation tests
cover the shared-arena path.

The full-run allocation profile fell from 1316.5 to 1177.6 MiB cumulatively.
Encoder preparation is 133.4 MiB and the single layer arena is 60.0 MiB. Output
remains byte-identical. Full synthesis took 92.26 s vs 91.80 s before; no speedup
is attributed to these memory changes. These allocation figures are not peak RSS.

`ExpF32To` adds checked AVX2/FMA exponential with scalar fallback outside [-32,32]
and for exceptional inputs. It accepts exact in-place operation, rejects partial
overlap, and allocates nothing. Tests cover a dense [-80,80] grid, random inputs,
NaN/infinities, signed zero, underflow/overflow, SIMD in-place lanes and tails.
Relative error is below 2e-6 in the tested finite corpus. An initial polynomial
FMA operand-order bug was caught by the tests and corrected before integration.
The 1024-element benchmark measured 13.75 us scalar vs 2.34 us dispatched.

`SoftmaxSIMDInPlace` uses that kernel in OmniVoice block attention; generic
`SoftmaxInPlace` is unchanged. Sequential float32 summation and exceptional-value
behaviour are preserved. Boundary and random-row tests pass, with 0 allocations.
128/218-element rows measured about 3.3x faster. Real model tests and the complete
WAV hash still pass. Full synthesis with SIMD softmax took 94.05 s in one run,
so there is no demonstrated end-to-end speedup. Profile and JSON:
`/workspace/tmp/omnivoice-full-native-v5.{cpu,mem,json}`; waveform
`/workspace/tmp/synthetic-spock-full-native-v5.wav` has the same SHA-256 as v2–v4.

Native tests/vet, race/no-CGo checks, and ARM64 CLI/test cross-builds pass.
Runtime feature detection requires both AVX2 and FMA before executing the exp
assembly. Other architectures currently use its scalar fallback.

## Native preprocessing, post-processing and SiLU (2026-09-13)

`-preprocess-reference` normalises original reference RMS, converts to PCM16 for
pydub-compatible silence detection, trims with mid/lead/trail 200/100/200 ms,
then encodes without a second RMS normalisation. Original RMS is retained for
output restoration. The approved chess reference now matches all 832 Python
preprocessed codes exactly (104 frames). Raw-reference mode still has 896 codes.

`-postprocess` trims generated silence with 500/100/100 ms, restores reference
RMS, applies 100 ms fades and pads 100 ms at each end. Both flags default off;
raw-output regression hashes therefore remain valid. Example:

```sh
bin/omnivoice -mode synthesize -model "$MODEL" -preprocess-reference -postprocess \
  -reference reference.wav -transcript 'Exact reference transcript.' \
  -text 'The evidence is insufficient, Captain.' -frames 75 -steps 8 \
  -output synthetic-processed.wav
```

Silence helpers match quantised PCM16 RMS, truncation, Python ties-to-even
millisecond rounding, 10 ms scans and overlapping keep-silence boundaries.
Synthetic Python parity covers short, long-middle, edge and all-silent audio,
plus fade/pad boundaries. Public duration/sample options reject oversized or
nonfinite values before conversion/allocation. `MaxSamples` bounds input;
post-padding output has a separate 20-second ceiling. Default fade/pad input
limit is 10 seconds (up to 10.2 seconds with default padding). Preprocessed
references shorter than two seconds are rejected by the current encoder limit.

`SiLUMulExpTo` combines SIMD exponential/vector multiply with scalar division and
uses tiled, idle GEMM packing scratch in the OmniVoice block. It adds no workspace
memory or allocations. Random [-100,100], exceptional, alias and tail tests pass;
FFN-sized microbenchmark measured 6.18 ms baseline vs 4.43 ms. The raw-reference
full run took 85.95 s versus 94.05 s previously; single-run timing, same WAV hash.

The combined processed native run took 83.29 s and produced 3.05 seconds of audio.
External Azure Speech recognition returned exactly “The evidence is insufficient,
Captain.” (confidence 0.914). ASR does not establish speaker similarity.
Private result `/workspace/tmp/synthetic-spock-native-processed-v1.wav`; timings
`/workspace/tmp/omnivoice-native-processed-v1.json`; ASR JSON
`/workspace/tmp/omnivoice-native-processed-asr.json`. Native tests/vet, Python audio
parity, race/no-CGo checks and ARM64 cross-builds pass. Listening acceptance has
not been recorded.

## Bounded long-utterance generation (2026-09-13)

`plan-chunks` writes private prepared chunk JSON without model inference.
`synthesize-long` prepares the same plan, loads the model/reference/decoder once,
then generates chunks sequentially using a shared streamed weight arena.
Per-chunk activation and logits setup still allocates. `-frames` is the maximum
per-chunk frame count in these modes, not the total output duration.

```sh
bin/omnivoice -mode plan-chunks -model "$MODEL" \
  -reference-tokens reference-codes.json -text 'A longer paragraph...' \
  -frames 100 -output private-chunk-plan.json
bin/omnivoice -mode synthesize-long -model "$MODEL" -postprocess \
  -reference-tokens reference-codes.json -text 'A longer paragraph...' \
  -frames 100 -steps 8 -output synthetic-long.wav
```

The planner uses upstream-style character weighting and short-duration boost,
with a native bounded partition policy: prefer punctuation, then words, then
safe rune boundaries. Actual tokenisation must fit 512 positions. Bracketed tags
and `<|...|>` control tokens cannot be split; oversized protected spans fail.
Concatenating chunk text exactly reconstructs trimmed input. Input is bounded to
16,000 runes, 128 chunks and 15,000 generated frames (10 minutes). UTF-8 errors
are rejected. Candidate prompts are not retained in a large cache. Estimates
are heuristic; they do not establish multilingual pronunciation quality.

Chunks retain the same reference and restart seed 42. Assembly applies 5 ms
edge fades and 100 ms gaps, even without `-postprocess`; with that flag, each
chunk also gets output silence trimming and reference loudness restoration.
It does not apply the single-shot 100 ms fades/padding to every chunk. This
policy is not upstream chunk-equivalent and can affect inter-chunk prosody.

Real checkpoint test: a five-sentence paragraph produced 13.42 seconds across
four chunks (342 target frames) in 344.46 seconds at eight steps. Azure Speech
recovered the entire paragraph exactly, including a word-boundary split inside
one sentence (confidence 0.842). Listening acceptance has not been recorded.
Private audio `/workspace/tmp/synthetic-spock-long-v1.wav`; metadata
`/workspace/tmp/omnivoice-long-v1.json`; ASR `/workspace/tmp/omnivoice-long-asr.json`.
Text/tag preservation, CJK boundaries, insufficient capacity, excessive chunks,
invalid UTF-8 and wave-assembly tests pass, including real-tokenizer planning.
Native tests/vet, race/no-CGo checks and ARM64 cross-build pass.

## Still required for completion

1. Reduce setup memory (loaded weights and conservatively sized scratch) and
   evaluate resident-service reuse; successful prepared reference calls allocate zero.
2. Broader multilingual/Unicode tokenizer parity beyond the current fixtures.
3. Remaining SIMD work: packing, SiLU division/GELU, sine and codec scatter;
   exponential/softmax SIMD currently accelerates amd64 only.
4. Text/reference → native speech listening acceptance against approved sample 3.
5. Broader long-utterance quality tests and listening evaluation of continuity
   at generated chunk boundaries; reduce per-chunk setup allocations.

The goal remains active. This is a working staged implementation, not a completed
fully native replacement or a full-SIMD graph.

## Multilingual tokenization and chunk workspace reuse (2026-09-13)

The 24 tokenizer fixtures cover accented/decomposed Latin, CJK, Arabic, Indic
scripts, emoji, digits and whitespace. The fixture generator preserves the source
NFC normalizer and pre-tokenizer configuration; expected IDs come from upstream
`tokenizers`. Both the reduced fixture and the full model tokenizer pass the same
cases. `golang.org/x/text/unicode/norm` handles NFC before ordinary BPE encoding;
special-token matching still happens first. This supports OmniVoice's NFC and
Qwen-style split configuration, not arbitrary Hugging Face normalizer pipelines.
Mixed normalizer sequences and standalone Split pre-tokenizers are not implemented.
The tests establish token IDs, not multilingual pronunciation quality.

Long synthesis constructs backbone, generation and decoder reservations once from
the maximum planned shape. `Reconfigure` changes active tensor views without
padding attention or allocating new inference workspaces. Reconfigure both
backbones before the generation workspace. The API rejects stale targets larger
than the active backbone. Generation resets the seed and mask state per chunk.
Post-processing, retained chunk copies and final WAV assembly still allocate.

Tests cover larger/smaller/larger shapes against independent runners, zero-allocation
backbone/generation resizing, decoder scratch reuse, cancellation during backbone
and decoder execution, and recovery after cancellation. The CLI passes SIGINT and
SIGTERM cancellation through generation and decoding. Partial results from a failed
call must be discarded; the same workspace can be retried.

The saved four-chunk 13.42-second sample took 335.17 seconds, compared with 344.46
seconds before workspace reuse. Both WAVs have SHA-256
`bd1f3ec5a2bc5ece889b53d2462e0ab782274aa0b7953ebd0090b03ee90b79f9`.
A single timing pair does not establish a speedup. The allocation profile totals
359.9 MiB, including tokenizer/model loading, post-processing and output storage.
This run used cached reference codes; native reference encoding was tested separately.

Affected tests, vet, race, no-CGo tests and the ARM64 CLI cross-build pass. The
real-checkpoint decoder still matches all 1,920 reference samples with maximum
absolute error 5.59e-7 and zero prepared `DecodeInto` allocations. The full repository
build still fails in the previously identified SpacemiT and DiffusionGemma packages.

## Tokenizer load memory reduction (2026-09-13)

Merge format selection now precedes decoding, avoiding the failed string-array
attempt for Qwen array-form merges. A sizing pass over validated JSON reserves the
merge slices once; decoding still uses `encoding/json`. Measured allocations fall
from 112.2 MiB to 60.2 MiB per real-tokenizer load. Token IDs are unchanged across
all 24 real-checkpoint fixtures. See [PROFILING.md](PROFILING.md) for the isolated
benchmark and validation. No new end-to-end synthesis measurement was made for
this loader-only change.

## Exact SIMD packing (2026-09-13)

On AVX2/FMA amd64 hosts, sixteen-row GEMM panels now use an exact four-column SIMD
transpose. Partial panels and column tails remain bounded; scalar fallback runs
when CPU features are disabled. This removes the scalar full-panel hotspot without
changing model arithmetic or adding allocations. The complete native WAV matches
the previous raw-reference baseline byte for byte. Direct panel benchmarks improve
3.5–7.9×; whole synthesis remains about 86 seconds in the measured run. See
[PROFILING.md](PROFILING.md) for details and limits. The GEMM microkernel is still the
largest CPU cost; a scheduling-only candidate failed to establish a gain and was
not enabled. Scalar sine, erf and some exponential/division paths still exist.

## Codec SIMD sine (2026-09-13)

Encoder and decoder Snake activations use a bounded AVX2/FMA sine approximation
on amd64, with scalar fallbacks and 256-value stack scratch. Prepared calls still
allocate zero bytes. All 832 preprocessed reference codes remain unchanged, and
real codec parity passes. The full WAV has only 61 one-step PCM16 differences
out of 72,000 samples. The channel benchmark is about 3.4× faster; whole synthesis
did not improve in the measured run. See [PROFILING.md](PROFILING.md) for numerical
bounds, artifacts and fallback coverage. HuBERT erf, encoder/sampler exponentials,
SiLU division and unsupported-architecture sine remain scalar.

## SIMD SiLU division (2026-09-13)

The amd64 SiLU finishing stage uses explicit vector add/divide/multiply, preserving
float32 rounding and exceptional-value classification. The full native WAV is
byte-identical to the preceding sine-enabled version and prepared calls remain
allocation-free. The representative FFN benchmark improves about 10–12%.
Non-amd64 finishing, HuBERT erf and encoder/sampler exponentials still use scalar
operations. See [PROFILING.md](PROFILING.md) for timings and validation.

## Semantic encoder SIMD ELU (2026-09-13)

The encoder uses a near-zero-safe expm1 kernel for bounded negative ELU inputs,
with exact positive/NaN passthrough and conservative scalar fallback regions.
Representative mixed-sign benchmarks improve about 3.2×; prepared calls allocate
zero bytes. Raw and preprocessed native reference codes are unchanged (896 and
832 codes respectively). See [PROFILING.md](PROFILING.md) for numerical limits and
fallback-heavy performance. HuBERT erf and sampler exponentials remain scalar.

## HuBERT SIMD erf-GELU (2026-09-13)

HuBERT's convolutional and FFN activations now use a bounded SIMD approximation
to erf-GELU on AVX2/FMA hosts. Scalar fallback covers tiny, exceptional and
out-of-range inputs. Five- and 100-frame real-checkpoint parity tests pass with
zero prepared allocations, and all raw/preprocessed reference codes remain
unchanged. Direct eligible-input benchmarks improve roughly 10×; fallback-heavy
inputs may be slower. Sampler exponentials and the documented architecture/range
fallbacks remain scalar. See [PROFILING.md](PROFILING.md) for bounds and evidence.

## Sampler SIMD exponentials (2026-09-13)

Conditional/unconditional guidance log-softmax uses existing scratch for bounded
SIMD exponentials and retains sequential float64 summation. Exceptional rows keep
the original scalar policy. Upstream fixture and full-generation checks pass; the
native sample is byte-identical. The isolated 1,025-class benchmark improves about
2.8× without allocations. Approximate probabilities may change near-tied selections
on other inputs. See [PROFILING.md](PROFILING.md) for limits and evidence.

## GEMM experiment and capability limits (2026-09-13)

An exact accumulator-preserving K-blocked GEMM candidate passed parity tests but
was neutral/slower than the current kernel on representative N100 projections.
It was not enabled. Representative projection benchmarks remain in the test suite.
Backend discovery now lists scalar fallback categories and whether approximate
nonlinear SIMD is active. The full graph is not SIMD-only; Vulkan still has no
native OmniVoice dispatch. See [PROFILING.md](PROFILING.md) for experiment results.

## Post-optimisation speech validation (2026-09-13)

The four-chunk paragraph was regenerated with the current kernels and the same
cached preprocessed reference, eight steps per chunk, 100-frame cap, five-ms
boundary fades and 100-ms gaps. It produced 13.42 seconds in 341.48 seconds.
Compared with the pre-nonlinear-optimisation long baseline, exactly 307 of 322,080
PCM16 samples differ, each by one integer step. No samples are clipped. External
Azure Speech recognition recovered the entire five-sentence paragraph exactly.
The run is not a demonstrated speedup over the earlier 335.17-second measurement.

A Portuguese sample used language `pt`, the same English reference/transcript,
eight steps and a 125-frame target with output post-processing:

> A evidência é insuficiente, capitão. Precisamos de investigar.

It produced 4.13 seconds in 114.22 seconds. No samples are clipped. Azure Speech
with `pt-PT` recovered the expected words, replacing the sentence break with a
comma. This tests one accented Portuguese sentence and cross-language reference
conditioning. It does not establish coverage for all supported languages or
regional pronunciation quality.

Validation artifacts (private local files, not distributed with source):

- `/workspace/tmp/omnivoice-long-optimized-v13.json`
- `/workspace/tmp/synthetic-spock-long-optimized-v13.wav`
- `/workspace/tmp/omnivoice-portuguese-v1.json`
- `/workspace/tmp/synthetic-spock-portuguese-v1.wav`
- `/workspace/tmp/omnivoice-quality-validation.json`
- `/workspace/tmp/omnivoice-quality-long-asr.json`
- `/workspace/tmp/omnivoice-quality-portuguese-asr.json`

Both synthetic WAVs were delivered for user listening review. Voice similarity,
Portuguese pronunciation and long-chunk transitions still require listening
acceptance. Assistant integration remains out of scope until that review. Current
performance is slower than real time, Vulkan inference is unimplemented, and
scalar range/architecture fallbacks remain. The measured optimisation round is
complete; subjective quality acceptance is not.
