# Qwen3-TTS ConvNeXt FC1 batched GPU stage check

A batched two-block NVIDIA diagnostic matched all **1,048,576** independent Rust/Candle F32 outputs of one released ConvNeXt FC1 stage **bitwise**. Ordinary full-width SGEMM missed the fixed `1e-6` stage gate by a maximum error of **2.67028809e-5**. No decoder waveform was computed with GPU FC1, and production synthesis stays CPU-only.

## Released stage and arithmetic

The opt-in `TestDecoderConvNeXtFC1PinnedGPU` verifies the approved Qwen3-TTS 0.6B CustomVoice speech-tokenizer config/weights, pinned 64-frame code fixture, and independent full Rust trace files before loading. `cn256` labels **256 time positions**, not 256 channels. The traced second pre-upsample ConvNeXt `pwconv1` maps `[256,1024]` to `[256,4096]` with bias. The trace input `cn256-norm.full.f32le` is 1,048,576 bytes, SHA-256 `0b45759f311819bf1cb04ddf91ac298adcde78d9c481d2cbd3b9d76db53a106f`; output `cn256-fc1.full.f32le` is 4,194,304 bytes, SHA-256 `04de2e4687fd288d6d8b59c6d420c59341ce4afa0e5fd87878ce6d666aac7aeb`.

The existing CPU `decoderLinearForward` matched the full Rust output bitwise. Ordinary GPU SGEMM over all 1,024 input channels differed from CPU and Rust at **945,417/1,048,576** positions, maximum absolute error **2.67028809e-5**. The test records this failure as diagnostic evidence but does not require it to persist if the backend changes. Candle-order CPU arithmetic reduces the 1,024 inputs in two 512-wide parts and adds their F32 results. Two GPU SGEMMs with the same packed partition, F32 partial order and CPU bias matched CPU and Rust **bitwise in ten ordinary and three isolated race runs**; the `1e-6` stage threshold was unchanged.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/rust \
  go test ./model/qwen3tts -run '^TestDecoderConvNeXtFC1PinnedGPU$' -count=10 -v -timeout=300s
```

The three diagnostic calls (one ordinary full-width, two blocked) used **3 launches, 35,651,584 uploaded bytes, 12,582,912 downloaded bytes** and matched allocation/free counts of **9 buffers / 48,234,496 bytes**. They allocate/free per call; this is not a resident-buffer or speed result. A GPU-wide `nvidia-smi` poll at a requested 100 ms interval observed **33–147 MiB across 55 samples**; it cannot resolve exact in-flight peak or attribute all memory to this test. Logs and polling data are under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/cn256-fc1-*`.

This test uses a saved Rust stage input. It does not establish a live ConvNeXt input boundary, downstream GELU/FC2 or final waveform parity when FC1 runs on GPU. A live chained test must retain the `1.6e-6` independent waveform gate, bounded request-local buffers, error cleanup and CPU default. Full GPU admission also needs the remaining decoder and Talker/CodePredictor, concurrent ownership, cancellation, actual peak/retained VRAM and full-request latency.
