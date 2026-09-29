# Nemotron 3 Diarization roadmap

NVIDIA's `nvidia/Nemotron-3-Diarization` is a candidate for a separate native
speaker-diarisation backend. An approved checkpoint has now run as an isolated
CPU reference on one 11-second fixture. Go-pherence has a bounded native frontend,
feature-stack projection and first-layer pre-attention projection, but no complete
Nemotron inference runtime. See [the bounded reference record](../validation/nemotron-speech-reference-2026-09-28.md).

## Pinned source and scope

- Model: [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/a435e9867d79e789e90053f9b6d6834053af564a), revision `a435e9867d79e789e90053f9b6d6834053af564a`.
- Published [configuration](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/a435e9867d79e789e90053f9b6d6834053af564a/config.json): `Nemotron3DiarizationForAudioFrameClassification`, 31 audio-transformer layers, width 512, 128 mel bins, factor-eight subsampling, and an eight-speaker head. The configuration also defines chunk/FIFO and speaker-cache state; their numerical and temporal behaviour has not been verified here.
- The [model card](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/a435e9867d79e789e90053f9b6d6834053af564a/README.md) describes offline and streaming diarisation for up to eight speakers. The Hub records `openmdw-1.1` licence metadata. Its OpenMDW-1.1 terms require retaining the licence and applicable copyright/origin notices on redistribution and leave third-party rights/permissions to the user; generated outputs have no licence-imposed obligations. Review deployment terms before checkpoint use.

The existing [Whisper and Community-1 workflow](speech-integration.md) keeps
plain transcript output independent of optional speaker attribution. Nemotron
would be a distinct opt-in speaker provider, not a silent replacement for
Community-1. Neither backend has trained-quality acceptance in the integrated
speech job.

## Pinned metadata inspection

At revision `a435e9867d79e789e90053f9b6d6834053af564a`, the public file list includes `config.json`, `processor_config.json`, `README.md`, an ASR integration guide, `.nemo` and safetensor checkpoints. The safetensors checkpoint was later fetched and integrity-checked after user approval; the isolated CPU reference run is recorded separately. The processor uses 16 kHz mono input, pre-emphasis 0.97, 512-point FFT, 400-sample window and 160-sample hop with 128 mel features. A factor-eight subsampler yields a nominal 80 ms model step; the card describes `(batch, frames, 8)` logits every 10 ms for its exposed frame classification, so timestamp alignment must be verified against an oracle rather than inferred from the subsampling factor alone.

The processor's default `low_latency` mode lists `[9,4]`; `very_low_latency` and `ultra_low_latency` list `[6,2]` and `[3,1]`. Model config has 31 attention layers of width 512, an eight-speaker head, `chunk_length: 340` and `chunk_right_context: 40`. It also distinguishes top-level FIFO/cache settings from a separate `streaming_config` (`fifo_length: 264`, `speaker_cache_length: 264`, update period 222). The card's Transformers streaming example passes `speaker_cache` between chunks and marks the last chunk explicitly. The exact units, mode-dependent lookahead and cache ownership require independent fixtures before a native streaming implementation.

## Proposed gates

1. Freeze the pinned configuration, processor contract, asset inventory,
   licences and file hashes. Check sample rate, feature framing, channel order,
   timestamp origin and maximum-speaker rules.
2. User approval now covers downloading needed Nemotron weights. Produce
   independently generated small fixtures for the frontend, subsampling,
   transformer/head, arrival-order speaker cache, streaming state and offline
   frame-to-turn decoding. Record the oracle software version, input hashes,
   output hashes, units and numerical tolerances.
3. Implement a scalar Go reference with bounded allocations and checked state
   transitions. Promote measured hot operations to checked SIMD backends only
   after operator and whole-recording parity. A Python/NeMo process is an oracle
   option, not the Go inference runtime.
4. Add an explicit speech-job provider selection while preserving plain
   transcript/VTT on diarisation failure. Test overlapping speakers, silence,
   unknown speakers, chunk seams, cancellation/retry, long recordings and
   deterministic turn labelling without reusing Community-1 checkpoints.
5. Measure diarisation error rate on an independently labelled cohort, both
   offline and streaming latency/throughput on each proposed native platform.
   Record speaker-count and recording-length limits. Do not infer accuracy from
   model metadata or synthetic operator tests.

The released checkpoint ran once through the pinned Transformers CPU reference.
The bounded 16-row `model/nemotrondiarization/Layer0QKV` path applies both
`audio_tower.input_layer_norm` and the first layer's `layer_norm1` before Q/K/V.
It matches independent PyTorch fixtures: maximum absolute error was
0.00000382 for Q, 0.00000286 for K and 0.000000834 for V on AVX2/FMA, with
zero values outside the documented absolute/relative tolerance. A one-CPU
microbenchmark of the 16-row operation took 0.396–0.490 ms, 163,840 B and
five allocations across five 20-iteration runs. Loading, frontend, stacking,
RoPE, attention and transfer costs are excluded. The 31-layer encoder, cache,
upsampler and speaker head have no native parity. `Layer0Attention.ForwardOffline`
passes PyTorch parity for an isolated unmasked 16-row bidirectional window:
maximum absolute error `5.72e-6` for the attention projection and `7.63e-6`
for its residual. Its one-CPU, five-by-20-iteration microbenchmark used
294,976 B/10 allocations and 0.877–2.306 ms per window; timings are noisy.
The window does not reproduce the first 16 rows of the full 138-row recording.
The full 138-row first-layer attention also matches the PyTorch reference on
JFK: max error `7.63e-6` for attention and `1.14e-5` after residual. The
processor marks 1100 of 1101 feature frames valid; sampling that mask every
eight frames marks all 138 stacked positions valid. The one-CPU 138-row
benchmark after sharing the owned input-normalisation buffer took 22.4–29.5 ms
and 2,294,336 B/9 allocations in five 10-iteration runs, down from
2,581,056 B/10 allocations. Timing samples overlap, so there is no speed
claim. The bounded complete first audio layer adds `layer_norm2`, the 512→2048→512
GELU MLP and its residual. Independent PyTorch 16- and 138-row outputs pass:
full-context maximum absolute error `1.53e-5`, mean `3.33e-7`, with zero
values outside the calibrated tolerance on AVX2/FMA. The scalar fallback
passes too. One-CPU 138-row samples took 43.1–44.3 ms with 3,998,272 B/12
allocations across five 10-iteration runs. Sharing each position's RoPE
sine/cosine across eight heads kept fixture parity; subsequent five
10-iteration runs took 41.8–43.9 ms for the full layer and 21.6–22.5 ms
for attention alone, with unchanged allocations. Timing samples overlap, so
this does not establish a speedup. Checked SGEMM for the two attention matrix
products passed the independent PyTorch fixture and scalar comparison for
windows of 1, 7, 15, 16, 17, 64, 138 and 376 rows. At 138 rows, three
20-iteration single-CPU samples took 7.03–8.55 ms for attention and
26.75–31.73 ms for the complete layer, versus the prior attention samples
of 21.6–22.5 ms. Scratch increased attention allocations from 2,294,336 B/9
to 2,539,520 B/13 (the complete layer uses 4,243,456 B/16 allocations).
These calls exclude loading, frontend,
stacking and the other 30 layers. The composed 138-row layer-0 output also passes the second layer's
pre-attention normalisation and Q/K/V fixture checks. On AVX2/FMA, maximum
absolute errors are `9.54e-7` for normalisation, `3.34e-6` for Q,
`2.86e-6` for K and `9.54e-7` for V, with zero values outside tolerance.
The second layer's full-window attention and residual now pass separate
16- and 138-row PyTorch fixtures via `Layer1Attention.ForwardOffline`. Both
checked SGEMM and scalar paths have zero values outside
`3e-4 + 2e-5*abs(reference)` and mean error below `2e-6`. The largest
residual error is `3.05e-5` on AVX2/FMA and `6.10e-5` with CPU features
disabled; 16-row and 138-row windows have different
bidirectional contexts. These tests include the composed first layer and exclude loading, frontend
and stacking. `Layer1Complete.ForwardOffline` also composes the second
layer's normalised GELU MLP and passes independent 16- and 138-row output
fixtures: AVX2/FMA maximum absolute error `2.29e-5`, scalar fallback
`3.81e-5`, mean below `1e-6`, and zero per-value outliers. The layer-1 math also loads an indexed layer from the released 31-layer
audio tower. A separately generated full-context 138-row PyTorch fixture
checks the composed first three layers through layer-2 normalisation,
attention, residual and complete GELU MLP: AVX2/FMA maximum output error
`2.29e-5`, scalar fallback `5.34e-5`, mean below `2e-6`, with zero
per-value outliers. `OfflineAudioTower.ForwardOffline` now composes all 31 released audio layers
and the tower's final LayerNorm over the same unmasked 138-row window. Pinned
PyTorch checkpoints after layers 7, 15, 23 and 30, plus the final norm,
pass AVX2/FMA and CPU-features-disabled checks with zero values outside
`3e-4 + 2e-5*abs(reference)`. The raw layer-30 output has AVX2/FMA maximum
error `8.55e-4`, mean `2.08e-5` (scalar maximum `1.10e-3`, mean `2.40e-5`);
its measured mean gate is `3e-5`. The final-normalised output has AVX2/FMA
maximum error `4.77e-6`, mean `5.66e-7` (scalar maximum `1.24e-5`, mean
`6.51e-7`), with a `2e-5` mean gate. These are one-recording, full-context
encoder measurements; they do not qualify arbitrary lengths, masks, chunking,
or concurrent serving. Frontend composition, cache, upsampler and speaker head
have no native end-to-end parity or full-request latency evidence. A separate `OfflineHead.ForwardOffline`
path now matches pinned PyTorch projection, 3-tap subpixel convolution,
upsampling and eight-speaker logits for the 138-row tower output. The isolated
head's logits have AVX2/FMA maximum error `1.53e-5`, mean `7.03e-7`;
scalar maximum `1.91e-5`, mean `1.45e-6`, with zero per-value outliers.
Composing the Go tower and head yields 1104×8 logits before trimming to
valid input frames: AVX2/FMA maximum error `1.91e-5`, mean `3.41e-6`;
scalar maximum `2.67e-5`, mean `3.67e-6`, with zero outliers. These
prepared-stacking windows isolate model numerics. `OfflineRequest.ForwardPCM`
now composes the native 16-kHz log-mel frontend, stack projection, 31-layer
tower and head on the pinned 176,000-sample JFK waveform. The owned 1101×8
raw logits match the independent PyTorch request: AVX2/FMA maximum absolute
error `3.25e-5`, mean `2.84e-6`; CPU-features-disabled maximum `6.11e-5`,
mean `5.87e-6`, with zero values outside `3e-4 + 2e-5*abs(reference)`.
The last masked feature row is retained and logits are trimmed from 1104 to
1101 frames, matching the reference request shape. `ExtractSegments` also matches the released processor's default 0.5 activity
threshold, 1100-valid-frame mask and two-decimal timestamps on this request:
speaker 0 spans `[0.28, 2.28]`, `[3.27, 4.56]`, and `[5.36, 10.63]`
seconds. A synthetic overlap/masked-frame check covers ordering and ownership.
This is processor-output parity for one recording, without labelled DER.
The head's subpixel convolution now packs three adjacent projected frames
into checked dense rows, with Conv1d weights reordered once at load time.
On the same i7-12700 with `GOMAXPROCS=1`, five 10-iteration isolated
138-row head samples fell from `133.09–134.77 ms` to `6.36–6.76 ms` per
call. Owned scratch increased from `3,555,328 B/6 allocations` to
`3,874,816 B/7 allocations`. PyTorch stage, composed-logit and PCM-request
fixtures pass SIMD and CPU-features-disabled gates with zero per-value
outliers; the changed reduction order raises the isolated convolution
maximum to `3.63e-5` on AVX2/FMA. These timings exclude loading, frontend,
tower, speaker extraction and transfers; no whole-request speedup has been
measured. The composed CPU path now uses the checked AVX2/FMA `GELUErfF32To` in each
layer's MLP, falling back to scalar exact-erf outside the kernel's range.
Released-weight layer checkpoints, full PCM logits and default-threshold
segments still pass on AVX2/FMA and with CPU features disabled. On the same
i7-12700 with `GOMAXPROCS=1`, five 5-iteration whole-request samples moved
from `856–864 ms` before the GELU change to `565–629 ms` after it, with
`479 allocations` and about `130.37 MB` per request; loading is excluded.
The two sample ranges do not overlap, but a longer controlled latency run
and independent quality evidence are needed before a production speed claim.
A separate `SpeakerFIFO` now matches the pinned streaming cache's
pre-compression embedding transitions for three prepared low-latency chunks
of nine current and four lookahead encoder frames: returned input lengths
13, 22 and 31; retained FIFO lengths 9, 18 and 27. Lookahead frames are
not retained. The released streaming cache uses a 264-frame FIFO and moves
at least 222 frames into a speaker cache on overflow; the separate offline
configuration uses a 40-frame FIFO and 300-frame update period. The Go
pre-compression slice rejects overflow without changing state. `PoolSpeakerProbabilities` also matches the reference's sigmoid and
8-logit average pool for all three prepared steps, with one masked final
encoder row: maximum absolute error `5.97e-8`, mean below `1.3e-8` and
zero values outside `2e-6`. It rejects non-finite logits even on masked
rows and preserves caller input. Prepared embeddings and synthetic logits
do not qualify speaker-cache scoring/compression, model inference,
streaming, other input lengths, transfer-inclusive GPU latency or
production readiness.
