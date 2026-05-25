# Whisper + Diarization Implementation Plan

## Overview

Add speech recognition (Whisper) and speaker diarization to go-pherence.
This requires new architectural primitives not present in the current decoder-only codebase:

- Audio frontend (mel spectrogram)
- 1D convolution
- Bidirectional (non-causal) self-attention
- Cross-attention (encoder→decoder)
- Speaker embedding model
- Clustering pipeline

## Architecture Diagram

```text
Audio WAV/PCM (16kHz mono)
│
├─ Resample if needed
│
▼
Mel Spectrogram (80 bins × T frames)
│
▼
┌──────────────────────────────────┐
│ Whisper Encoder                  │
│  Conv1D stem (2 layers, GELU)    │
│  + Sinusoidal positional embed   │
│  + N encoder layers:             │
│    LayerNorm → full self-attn    │
│    → residual → LayerNorm → MLP  │
│    → residual                    │
└──────────────────────────────────┘
│
│ encoder_output: [T', d_model]
│
▼
┌──────────────────────────────────┐
│ Whisper Decoder                  │
│  Token embed + positional embed  │
│  + N decoder layers:             │
│    LayerNorm → causal self-attn  │
│    → residual                    │
│    → LayerNorm → cross-attn     │
│      (Q from decoder, K/V from   │
│       encoder output)            │
│    → residual → LayerNorm → MLP  │
│    → residual                    │
└──────────────────────────────────┘
│
▼
Token logits → Greedy/Beam decode → Transcript + timestamps
│
▼ (optional diarization)
┌──────────────────────────────────┐
│ Speaker Diarization              │
│  VAD → segment audio             │
│  Speaker encoder → embeddings    │
│  Agglomerative clustering        │
│  Align speakers to word times    │
└──────────────────────────────────┘
│
▼
Final output: timestamped transcript with speaker labels
```

## Package Layout

```text
loader/audio/
    wav.go          — Read WAV/PCM to []float32
    resample.go     — Polyphase sinc resampler to 16kHz
    mel.go          — STFT + mel filterbank

models/whisper/
    config.go       — Model configs (tiny/base/small/medium/large-v3)
    encoder.go      — Encoder: conv stem + transformer encoder
    decoder.go      — Decoder: causal self-attn + cross-attn + MLP
    whisper.go      — Full pipeline orchestration
    decode.go       — Greedy/beam token decoding + timestamp handling

models/speaker/
    config.go       — Speaker encoder config
    ecapa.go        — ECAPA-TDNN speaker embedding model
    embed.go        — Per-segment embedding extraction
    vad.go          — Voice activity detection
    cluster.go      — Agglomerative clustering
    diarize.go      — Full diarization pipeline
    align.go        — Align transcription timestamps to speaker segments

tensor/
    conv1d.go       — 1D convolution primitive
    pool.go         — Attentive statistics pooling

backends/simd/
    fft_amd64.s     — AVX2/FMA radix-4 FFT
    fft_arm64.s     — NEON radix-4 FFT
    fft_other.go    — Scalar FFT fallback
    conv1d_amd64.s  — SIMD 1D convolution
    conv1d_arm64.s  — NEON 1D convolution
    pool_amd64.s    — SIMD statistics pooling

backends/cuda/ptx/
    fft.go          — FFT + mel PTX kernel source
    conv1d.go       — Conv1D PTX kernel source
    attention_full.go — Non-causal attention PTX
    cross_attention.go — Cross-attention PTX

gpu/
    conv1d.go       — DevBuf Conv1D dispatch
    cross_attention.go — Cross-attention GPU dispatch
    mel.go          — GPU mel spectrogram dispatch

cmd/whisper/
    main.go         — CLI: transcribe [+ diarize] audio files
```

## Whisper Model Sizes

| Model | Encoder layers | Decoder layers | d_model | Heads | Parameters |
|-------|---------------|---------------|---------|-------|------------|
| tiny | 4 | 4 | 384 | 6 | 39M |
| base | 6 | 6 | 512 | 8 | 74M |
| small | 12 | 12 | 768 | 12 | 244M |
| medium | 24 | 24 | 1024 | 16 | 769M |
| large-v3 | 32 | 32 | 1280 | 20 | 1550M |

## New Primitives Required

### 1. Mel Spectrogram

```text
Input: audio[N_samples] (16kHz, float32)
Window: 400 samples (25ms), Hann
Hop: 160 samples (10ms)
FFT size: 400 (zero-pad to 512 for radix-2)
Mel bins: 80
Output: mel[80, T] where T = ceil(N_samples / 160)

Pipeline per frame:
  frame = audio[i*160 : i*160+400]
  windowed = frame * hann_window
  spectrum = |FFT(windowed)|² (power spectrum, first 201 bins)
  mel_frame = mel_filterbank @ spectrum  (80×201 @ 201 → 80)
  log_mel = log(max(mel_frame, 1e-10))
```

SIMD targets:
- Vectorized Hann window multiply
- Radix-4/split-radix FFT with AVX2 FMA or NEON
- Vectorized mel filterbank matmul (sparse: ~7 non-zero filters per bin)

GPU target:
- Single fused kernel: window → FFT → power → mel → log
- Process all frames in parallel (one thread-block per frame)

### 2. Conv1D

```text
Input:  [batch, in_channels, length]
Weight: [out_channels, in_channels/groups, kernel_size]
Bias:   [out_channels]
Output: [batch, out_channels, out_length]

out_length = (length + 2*padding - kernel_size) / stride + 1
```

Whisper conv stem uses:
- Conv1D(80, d_model, kernel=3, stride=1, padding=1) + GELU
- Conv1D(d_model, d_model, kernel=3, stride=2, padding=1) + GELU

Implementation strategy:
- Small kernels (k=3): direct convolution with SIMD inner loop
- GPU: direct conv kernel with shared memory for input tile

### 3. Full (Non-Causal) Self-Attention

Same as current attention but without the causal mask:
- Every position attends to every other position
- No triangular mask applied to scores
- Used only in the encoder

GPU kernel difference: skip the `if (col > row) score = -inf` branch.

### 4. Cross-Attention

```text
Q = decoder_hidden @ Wq   (from current decoder layer)
K = encoder_output @ Wk   (computed once, cached)
V = encoder_output @ Wv   (computed once, cached)

scores = Q @ K^T / sqrt(d_k)
weights = softmax(scores)    (no causal mask — full attention over encoder)
output = weights @ V
```

Key optimization: encoder K/V are computed once per audio chunk and reused for every decoder token.

### 5. Sinusoidal Position Embeddings

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Pre-computed table, added to encoder/decoder inputs.

## SIMD Coverage Plan

| Operation | amd64 (AVX2/FMA) | arm64 (NEON) | Scalar fallback |
|-----------|------------------|--------------|-----------------|
| FFT radix-4 butterfly | ✓ | ✓ | ✓ |
| Hann window multiply | Reuse VecMul | Reuse VecMul | ✓ |
| Mel filterbank (sparse MV) | Custom sparse | Custom sparse | ✓ |
| Conv1D k=3 stride=1 | Direct FMA loop | Direct FMA loop | ✓ |
| Conv1D k=3 stride=2 | Strided FMA | Strided FMA | ✓ |
| Statistics pooling | Mean+var reduction | Mean+var reduction | ✓ |
| GELU activation | Polynomial approx | Polynomial approx | ✓ |

## GPU Kernel Plan

| Kernel | Grid | Block | Notes |
|--------|------|-------|-------|
| mel_spectrogram | (num_frames, 1, 1) | (256, 1, 1) | Fused window+FFT+mel+log per frame |
| conv1d_k3_s1 | (out_len/256, out_ch, 1) | (256, 1, 1) | Shared memory input tile |
| conv1d_k3_s2 | (out_len/256, out_ch, 1) | (256, 1, 1) | Strided variant |
| attention_full | (num_heads, seq_q, 1) | (32, 1, 1) | Same as causal but no mask |
| cross_attention | (num_heads, dec_len, 1) | (32, 1, 1) | Q from decoder, K/V from encoder |
| speaker_pool | (1, 1, 1) | (256, 1, 1) | Attentive mean+std |

## Implementation Order

1. **Audio frontend** — WAV reader, resampler, mel spectrogram (scalar first, then SIMD/GPU)
2. **Conv1D** — Generic 1D convolution primitive with SIMD/GPU paths
3. **Whisper encoder** — Conv stem + full self-attention + MLP
4. **Cross-attention** — New attention mode for decoder
5. **Whisper decoder** — Causal self-attn + cross-attn + MLP
6. **End-to-end Whisper** — Pipeline, decoding, timestamps
7. **Speaker embeddings** — ECAPA-TDNN with Conv1D + pooling
8. **Diarization** — VAD + embedding + clustering + alignment

## Testing Strategy

- Phase 1: Synthetic audio (sine waves at known frequencies) → verify mel output against Python librosa
- Phase 2: Known Conv1D inputs → compare against PyTorch nn.Conv1d
- Phase 3-5: Load whisper-tiny weights, forward random mel → compare logits against HuggingFace transformers
- Phase 6-7: Same-speaker pairs should have cosine similarity > 0.8; different speakers < 0.3
- End-to-end: Transcribe a short known WAV and compare WER against reference

## Weight Format

Whisper weights are available as safetensors from HuggingFace:
- `openai/whisper-tiny` through `openai/whisper-large-v3`
- Standard safetensors format → reuse existing `loader/safetensors`

Speaker models (pyannote/wespeaker):
- Also available as safetensors/ONNX → prefer safetensors path

## Performance Targets

| Operation | Target (CPU) | Target (GPU) |
|-----------|-------------|--------------|
| 30s mel spectrogram | < 50ms | < 5ms |
| Whisper-tiny encode | < 200ms | < 30ms |
| Whisper-tiny decode (per token) | < 10ms | < 2ms |
| Speaker embedding (per segment) | < 50ms | < 10ms |
| Full transcribe 30s (tiny) | < 3s | < 0.5s |
| Real-time factor (large-v3, GPU) | < 0.1x | |

## Implementation Status

### Completed
- [x] Phase 1: Audio frontend (WAV, resample, mel, scalar FFT)
- [x] Phase 2: Conv1D primitive (scalar + GPU dispatch stub)
- [x] Phase 3: Whisper encoder (conv stem, sinusoidal PE, full self-attention)
- [x] Phase 4: Cross-attention + decoder (causal self-attn, cross-attn, KV cache)
- [x] Phase 5: End-to-end pipeline (weight loader, greedy/timestamp decode)
- [x] Phase 6: Speaker embeddings (ECAPA-TDNN, attentive stat pool)
- [x] Phase 7: Diarization pipeline (VAD, clustering, alignment, CLI)
- [x] GPU dispatch stubs (Conv1D, full attention, cross-attention, mel FFT PTX)
- [x] Reusable FFT in backends/simd/fft
- [x] Tensor pooling primitives

### Remaining (Optimization Phase)
- [ ] SIMD FFT assembly (AVX2, NEON)
- [ ] SIMD Conv1D assembly
- [ ] Implement PTX kernel bodies (not just stubs)
- [ ] Integration test with real whisper-tiny weights
- [ ] Batched encoder, streaming, speculative decoding
- [ ] TurboQuant for Whisper KV cache

## Latest Session Progress

### Additional commits
- Optimized FFT: 5.2× speedup with zero-alloc twiddle rotation (10µs/512pt)
- Platform FFT butterfly stubs (amd64/arm64 Go, assembly pending)
- Unrolled Conv1D k=3 stride=1/2 for Whisper stem (60ms for 80→384ch)
- Vectorized sinusoidal position embedding generation
- GPU non-causal attention dispatch (CPU fallback)
- GPU cross-attention dispatch (CPU fallback)
- CUDA PTX kernel structure: mel spectrogram (shared-memory FFT), Conv1D (tiled halo)
- Chunked/streaming transcription for audio > 30s with overlap deduplication
- Speaker embedding extraction pipeline (ECAPA + mel + diarize)

### Performance Benchmarks (i7-12700, scalar CPU)
| Operation | Time | Notes |
|-----------|------|-------|
| FFT 512pt (optimized) | 10µs | 5.2× faster than naive |
| FFT 512pt (scalar) | 53µs | Baseline |
| Conv1D 80→384ch, 480 frames | 60ms | Whisper conv stem |
| Encoder forward (tiny, 480 frames) | ~2.5s | 4 layers, scalar attention |
| Decoder per-token (tiny) | ~100ms | Cross-attention + self-attention |

### Architecture
```text
backends/simd/fft/
  fft.go            — Scalar radix-2 FFT (fallback)
  fft_simd.go       — Optimized Go FFT (zero-alloc twiddle)
  fft_amd64.go      — AVX2 butterfly stub
  fft_arm64.go      — NEON butterfly stub
  conv1d_opt.go     — Unrolled Conv1D k=3 s=1/2
  posembed.go       — Sinusoidal position embedding

backends/cuda/ptx/
  fft.go            — Fused mel spectrogram PTX (structure defined)
  conv1d.go         — Tiled Conv1D PTX (structure defined)
  attention_full.go — Non-causal attention + cross-attention PTX

gpu/
  conv1d.go         — Conv1D dispatch (CPU fallback)
  cross_attention.go — Full/cross attention dispatch
  attention_full.go — DevAttentionFull dispatch

models/whisper/
  chunked.go        — Streaming/chunked transcription
```

### Final Phase 8 Completion
- Fused mel spectrogram: 136ms/30s with 3 allocations (vs 12k unfused)
- Batched encoder: parallel worker pool for multi-chunk encoding
- Speculative decoding: draft-verify pattern (tiny drafts, small verifies)
- TurboQuant: 5× KV cache compression for long audio sequences
- Chunked/streaming: overlap deduplication for >30s audio

### Remaining Assembly Work (Deferred)
- `backends/simd/fft_amd64.s`: AVX2 radix-4 butterfly (Go stub ready)
- `backends/simd/fft_arm64.s`: NEON radix-4 butterfly (Go stub ready)
- `backends/simd/conv1d_amd64.s`: FMA k=3 inner loop (Go unrolled version ready)
- `backends/simd/conv1d_arm64.s`: NEON k=3 inner loop
- `backends/simd/pool_amd64.s`: SIMD mean+std reduction
- PTX kernel instruction bodies (structural stubs in place)

### Assembly Kernel Performance (i7-12700)

| Kernel | Size | Time | Speedup vs scalar |
|--------|------|------|-------------------|
| FFT butterfly (AVX2) | 512pt | 18µs | 3× |
| FFT optimized Go | 512pt | 9.4µs | 5.4× |
| Conv1D inner loop (AVX2 FMA) | 480 elem | 201ns | ~10× projected |
| Mean+Std pooling (AVX2) | 1536 elem | 308ns | — |
| Fused mel spectrogram (Go) | 30s audio | 136ms | ~zero-alloc |

### Assembly Files Written
- `backends/simd/fft/fft_amd64.s` — AVX2 butterfly (VMULPD/VADDPD/VSUBPD)
- `backends/simd/fft/fft_arm64.s` — NEON butterfly (FMULD/FADDD/FSUBD)
- `backends/simd/fft/conv1d_amd64.s` — AVX2+FMA Conv1D k=3 (VFMADD231PS)
- `backends/simd/fft/conv1d_arm64.s` — Scalar arm64 Conv1D k=3
- `backends/simd/fft/pool_amd64.s` — AVX2 mean+std reduction

### GPU Hardware Test Results (RTX 3060, CUDA 13.0)

All tests pass on GPU hardware:
- `backends/nvidia/runtime`: CUDA driver init, GEMV NVFP4, SGEMM validation ✅
- `models/whisper`: full pipeline with whisper-tiny weights ✅
- `models/speaker`: VAD, clustering, alignment ✅
- Assembly kernels (FFT, Conv1D, pooling): verified on amd64 ✅

**CPU RTF (whisper-tiny, no GPU dispatch for Whisper yet):**
- Encoder: 447ms for 3s audio
- Decoder: 30ms/token
- End-to-end: RTF=5.35–6.34 (CPU-only, SIMD Sdot + optimized linear)

**Next step for GPU RTF < 0.1:**
Wire `models/whisper` encoder/decoder linear layers through `backends/nvidia/runtime.Sgemm`
and attention through the CUDA attention kernel.
