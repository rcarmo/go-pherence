# Qwen3-TTS live ConvNeXt FC1 GPU waveform check

A request-local two-block NVIDIA FC1 calculation in the live 64-frame CPU decoder passed the independent Rust/Candle waveform gate: maximum absolute error **1.28487591e-6** across 122,880 samples, below the unchanged **1.6e-6** limit. The GPU FC1 output matched the live CPU calculation **bitwise** for all 1,048,576 values. Production synthesis remains CPU-only.

The opt-in `TestDecoderConvNeXtFC1LiveGPUWaveform` uses the approved 0.6B CustomVoice speech-tokenizer config/weights, the hash-pinned 64-frame codes and waveform, and independent full Rust `cn256-norm` and `cn256-fc1` traces. It calls the second pre-upsample ConvNeXt FC1 on its live CPU-produced input. The new internal callback is nil in normal decoder calls; no saved Rust tensor is injected. The [saved-input stage check](qwen3-tts-gpu-convnext-fc1-20260927.md) separately established bitwise FC1 parity on Rust input with the same two 512-wide reductions.

| Live boundary against saved Rust F32 | Maximum absolute difference | Different F32 bits |
|---|---:|---:|
| ConvNeXt norm input, 262,144 values | 2.32458115e-6 | 250,430 |
| GPU FC1 output, 1,048,576 values | 3.05175781e-5 | 980,291 |
| GPU FC1 versus live CPU FC1, 1,048,576 values | 0 | 0 |
| Final clamped waveform versus independent Rust, 122,880 samples | 1.28487591e-6 | not counted |

The test checks live input drift under `4e-6`, live GPU/CPU FC1 bitwise, GPU output drift under `4e-5`, and the unchanged waveform gate independently. The input and output do not match the saved Rust intermediates bitwise. The waveform result does not establish bitwise end-to-end parity.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR=/workspace/tmp/qwen3tts-cap64-20260926/reference1 \
  GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/rust \
  go test ./model/qwen3tts -run '^TestDecoderConvNeXtFC1LiveGPUWaveform$' -count=3 -v -timeout=300s
```

Three ordinary runs and one isolated race run produced the same boundary and waveform results. The released CPU 64-frame sentence test also passed with maximum error **1.2848759070038795e-6**. The prepared GPU FC1 call launched two SGEMMs, uploaded two activation blocks, downloaded two partial outputs and made no in-request device allocations/frees. Six buffers allocated and freed **26,214,400 bytes** across setup/teardown. CUDA free-memory snapshots in a passing run were 12,332,695,552 bytes before setup, 12,305,432,576 after setup/request, and 12,332,695,552 after teardown. A separate GPU-wide `nvidia-smi` poll at a requested 100 ms interval observed **33–184 MiB over 284 samples**; this is neither an exact peak nor a per-request VRAM attribution. Raw results are under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/cn256-fc1-live-*`.

This diagnostic leaves all other decoder stages and Talker/CodePredictor on CPU. It has no full-request latency, concurrent GPU ownership, cancellation or retained/peak VRAM admission evidence. No production GPU dispatch or service state changed.
