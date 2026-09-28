# Qwen3-TTS four-stage live decoder GPU diagnostic

Three request-local GPU convolutions and one batched ConvNeXt FC1 in a live 64-frame decoder passed the unchanged independent Rust/Candle waveform gate. Maximum absolute waveform error was **1.28522515e-6** across 122,880 samples, below **1.6e-6**. Production synthesis remains CPU-only; other decoder stages and Talker/CodePredictor still run on CPU.

The opt-in `TestDecoderFourGPUStagesPinnedWaveform` verifies the approved CustomVoice decoder checkpoint config/weights, pinned 64-frame codes and waveform, independently traced pre-convolution and decoder-initial inputs/outputs, full ConvNeXt norm/FC1 tensors, and full final-Snake/final-convolution tensors. Every GPU callback uses the live upstream calculation. The test never substitutes a frozen Rust intermediate, and the default decoder supplies nil callbacks.

| Live output/input against pinned Rust F32 | Maximum absolute error | Different bits |
|---|---:|---:|
| Pre-convolution input and GPU output | 0 | 0 |
| ConvNeXt norm input, 262,144 values | 2.32458115e-6 | 250,430 |
| ConvNeXt GPU FC1, 1,048,576 values | 3.05175781e-5 | 980,291 |
| Decoder-initial input, 262,144 values | 3.02791595e-5 | 253,751 |
| Decoder-initial GPU output, 393,216 values | 6.7949295e-6 | 366,996 |
| Final-Snake output, 11,796,480 values | 7.95125961e-4 | 11,354,301 |
| Final GPU convolution output, 122,880 values | 1.28522515e-6 | 117,778 |

The pre-convolution, ConvNeXt FC1, and decoder-initial GPU outputs matched their **live CPU counterparts bitwise**. The final GPU convolution differed from live CPU at 56,103 positions, maximum absolute error **5.96046448e-8**. Intermediate Rust drift and live CPU/GPU comparisons are separately gated; the passing waveform is not bitwise end-to-end parity.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR=/workspace/tmp/qwen3tts-cap64-20260926/reference1 \
  GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/rust \
  go test ./model/qwen3tts -run '^TestDecoderFourGPUStagesPinnedWaveform$' -count=3 -v -timeout=300s
```

Three ordinary repeats and one isolated race run passed with the same numerical results. Backend counters during one diagnostic decode recorded **1,939 launches, 1,939 activation uploads, 1,939 partial downloads and zero per-request GPU allocations/frees**. Setup and teardown each accounted for **63 buffers / 107,433,600 bytes**. One ordinary CUDA free-memory snapshot series measured 12,332,695,552 bytes before preparation, 12,208,963,584 after preparation and 12,208,898,048 after the request, then 12,332,630,016 after teardown. A separate 100 ms requested GPU-wide poll observed **33–350 MiB across 301 samples**; this is not exact peak or per-request attribution. The isolated race run had a separate 78.6 MB change in free-memory snapshots during the request despite balanced buffer counters, so those counters and snapshots cannot establish peak or retained VRAM. Raw logs/polling series are under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/fourstage-*`.

No full-request latency or throughput comparison, concurrent GPU ownership, cancellation or sustained/peak VRAM admission has been completed. The [three-convolution decoder timing](qwen3-tts-gpu-threeconv-decoder-benchmark-20260927.md) was near tied with CPU on this input; FC1 integration adds an unmeasured cost and must not be inferred to improve decoder speed. No production GPU dispatch or service state changed.
