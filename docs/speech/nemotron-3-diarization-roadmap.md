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
benchmark took 22.2–27.8 ms and 2,581,056 B/10 allocations in five
10-iteration runs. Later MLP, encoder layers, cache, upsampler and speaker
head remain unqualified.
Labelled diarisation quality, production latency and readiness promotion remain open.
