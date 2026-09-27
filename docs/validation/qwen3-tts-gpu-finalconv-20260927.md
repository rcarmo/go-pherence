# Qwen3-TTS final decoder convolution GPU diagnostic

An opt-in GPU diagnostic matched the pinned Rust/Candle output of the 64-frame final decoder convolution within **1.04308128e-7** maximum absolute error. The GPU output differed from the live CPU convolution at 56,173 of 122,880 samples, with maximum absolute difference **5.96046448e-8**. This is one stage on a saved Rust input; production waveform generation remains CPU-only.

## Frozen stage and execution

The approved `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` checkpoint is loaded from `GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR`. The test checks the speech-tokenizer config/model hashes and the pinned 64-frame code fixture. Independent Rust/Candle decoder traces from the same 64-frame request are supplied separately with `GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR`: `finalsnake.full.f32le` (47,185,920 bytes; SHA-256 `24230d220270640cfc161ffc90b1671111d12d4cbec823cefafa70e8107fda1d`) and `finalconv.full.f32le` (491,520 bytes; SHA-256 `e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb`). The [released 64-frame CPU waveform check](qwen3-tts-cap64-parity-20260926.md) and [two-convolution hybrid check](qwen3-tts-gpu-decoder-hybrid-20260927.md) have separate end-to-end gates.

The final convolution has input `[96,122880]`, 7 taps, and one output channel. The CPU path's maximum difference from the Rust stage output was **1.04308128e-7**. The diagnostic uses two 336-wide SGEMMs per 128-position tile, with host im2col, activation uploads, partial downloads, ordered F32 accumulation, and CPU bias. Its **1e-6 stage gate** applies separately to GPU/CPU and GPU/Rust maximum absolute differences. The two GPU reductions do not reproduce every CPU F32 bit; no bitwise parity is asserted.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/rust \
  go test ./model/qwen3tts -run '^TestDecoderFinalConvPinnedGPU$' -count=10 -v -timeout=300s
```

Ten ordinary runs and three isolated race runs reproduced the numerical result. The diagnostic checked **1,920 launches, 1,920 activation uploads and 1,920 partial downloads**, with no per-tile device allocations or frees. Six prepared buffers allocated and freed **347,776 bytes** in all. This is a stage-correctness and buffer-lifecycle test, not a stage-speed comparison or an in-flight peak/retained VRAM measurement. The saved input does not establish agreement for the live final-convolution input from the Go decoder, and the test does not measure full-waveform correctness when this GPU stage is inserted. Cancellation, concurrent ownership, remaining decoder stages, Talker/CodePredictor and full-request latency still need qualification before any GPU dispatch.
