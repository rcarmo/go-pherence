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

## Still required for completion

1. Reduce setup memory (loaded weights and conservatively sized scratch) and
   evaluate resident-service reuse; successful prepared reference calls allocate zero.
2. Broader multilingual/Unicode tokenizer parity beyond the current fixtures.
3. Remaining SIMD work: packing, exponential/sine kernels and codec scatter.
4. Text/reference → native speech listening acceptance against approved sample 3.
5. Broader shape/error tests, long utterances/chunking and optional upstream
   post-processing (silence removal, fades/padding).

The goal remains active. This is a working staged implementation, not a completed
fully native replacement or a full-SIMD graph.
