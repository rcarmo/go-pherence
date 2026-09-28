# Qwen3-TTS live final-convolution GPU waveform check

A request-local GPU final convolution on the live 64-frame CPU decoder input passed the unchanged independent Rust/Candle waveform gate: maximum absolute error **1.28522515e-6**, below **1.6e-6** for 122,880 samples. Production synthesis still uses the CPU final convolution. The test substitutes no saved intermediate tensor.

## Boundaries and inputs

The opt-in `TestDecoderFinalConvLiveGPUWaveform` decodes the 64-frame pinned code fixture from the approved `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` checkpoint. It hash-checks the speech-tokenizer weights/config, pinned code fixture, independent waveform, and full Rust/Candle final-Snake input and final-convolution output traces. The final convolution alone uses the test-only NVIDIA callback; all upstream decoder stages run on CPU. `decodeCodes` continues to select CPU convolutions by default.

| Boundary on live request | Maximum absolute difference | Different F32 bits |
|---|---:|---:|
| Final-Snake output versus saved Rust input, 11,796,480 values | 7.95125961e-4 | 11,354,301 |
| GPU versus live CPU final convolution, 122,880 values | 5.96046448e-8 | 56,103 |
| GPU final convolution versus saved Rust output, 122,880 values | 1.28522515e-6 | not counted |
| Clamped GPU waveform versus independent Rust waveform, 122,880 samples | 1.28522515e-6 | not counted |

The live-input comparison has an explicit `1e-3` drift bound, GPU/live-CPU output has a `1e-6` stage bound, and GPU/Rust output and waveform retain the `1.6e-6` bound. These are different checks: the upstream final-Snake tensor differs from Rust, and neither the final convolution nor the full waveform is bitwise end-to-end parity. The [saved-input final-convolution probe](qwen3-tts-gpu-finalconv-20260927.md) checks a separate frozen boundary.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR=/workspace/tmp/qwen3tts-cap64-20260926/reference1 \
  GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/rust \
  go test ./model/qwen3tts -run '^TestDecoderFinalConvLiveGPUWaveform$' -count=3 -v -timeout=300s
```

Three ordinary runs and one isolated race run passed with the same numerical results. The released CPU sentence test still passed with maximum waveform error **1.2848759070038795e-6**. During the diagnostic decode, backend counters recorded **1,920 launches, 1,920 activation uploads and 1,920 partial downloads**, with zero per-request GPU allocations/frees after buffer setup. Six buffers allocated and freed **347,776 bytes** in all. CUDA free-memory snapshots on one run were 12,332,695,552 bytes before setup, 12,330,598,400 after setup and request, and 12,332,695,552 after teardown. A separate 100 ms requested `nvidia-smi` poll returned **291 GPU-wide samples**, with **33 MiB idle and 149 MiB highest observed**. The poll may miss brief spikes and cannot attribute usage to this request; snapshots and balanced counters do not measure exact peak or long-term retention. Raw results are under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/finalconv-live-*`.

This diagnostic does not time the full request against a CPU baseline. The remaining decoder, Talker/CodePredictor, cancellation, concurrent ownership, sustained/peak VRAM and full-request latency are unqualified. No production GPU dispatch or service state changed.
