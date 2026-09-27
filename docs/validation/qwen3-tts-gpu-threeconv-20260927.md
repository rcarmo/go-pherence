# Qwen3-TTS three-convolution live decoder GPU diagnostic

Three request-local GPU convolutions in one live 64-frame decoder run passed the unchanged independent Rust/Candle waveform gate: maximum absolute error **1.28522515e-6** for 122,880 samples, below **1.6e-6**. All other decoder stages ran on CPU. Production synthesis remains CPU-only.

The opt-in `TestDecoderThreeGPUConvsPinnedWaveform` checks the approved CustomVoice speech-tokenizer weights/config, the frozen 64-frame code and waveform fixtures, two pinned convolution traces and the independent full Rust final-convolution traces before loading. The live decoder passes its own tensors to the NVIDIA callbacks; it does not inject a Rust intermediate. The [two-convolution check](qwen3-tts-gpu-decoder-hybrid-20260927.md) and [live final-convolution check](qwen3-tts-gpu-finalconv-live-20260927.md) qualify the stages separately.

| Live boundary against pinned Rust F32 | Maximum absolute error | Different bits |
|---|---:|---:|
| Pre-convolution input, 32,768 values | 0 | 0 |
| Pre-convolution GPU output, 65,536 values | 0 | 0 |
| Decoder-initial input, 262,144 values | 3.02791595e-5 | 253,751 |
| Decoder-initial GPU output, 393,216 values | 6.7949295e-6 | 366,996 |
| Final-Snake output / final-convolution input, 11,796,480 values | 7.95125961e-4 | 11,354,301 |
| Final-convolution GPU output, 122,880 values | 1.28522515e-6 | 117,778 |

The pre-convolution and decoder-initial GPU outputs match their **live CPU calculations bitwise**. The final GPU convolution differs from live CPU at 56,103 of 122,880 positions, with maximum absolute error **5.96046448e-8**. These are distinct from the Rust boundary comparisons; the run does not establish bitwise end-to-end parity. It checks the first Rust boundaries bitwise, decoder-initial input/output drift under `4e-5`/`1e-5`, final input under `1e-3`, and final output and waveform under the unchanged `1.6e-6` gate.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR=/workspace/tmp/qwen3tts-cap64-20260926/reference1 \
  GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/rust \
  go test ./model/qwen3tts -run '^TestDecoderThreeGPUConvsPinnedWaveform$' -count=3 -v -timeout=300s
```

Three ordinary repeats and an isolated race run gave the same values. During the diagnostic decode, backend counters recorded **1,937 launches, 1,937 activation uploads, 1,937 partial downloads and zero GPU buffer allocations/frees** after preparation. Setup and teardown each accounted for **57 buffers / 81,219,200 bytes**. CUDA free-memory snapshots on one run were 12,332,695,552 bytes before setup, 12,236,226,560 after setup and request, and 12,332,695,552 after teardown. In one repeat run, a GPU-wide `nvidia-smi` poll at a requested 100 ms interval returned 299 samples with **33 MiB idle and 239 MiB highest observed**. Sampling may miss brief peaks and does not attribute usage to this request; balanced allocation counters do not establish long-term retention. Raw logs and sampled series are under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/threeconv-*`.

No full-request CPU/GPU latency comparison, cancellation, concurrent GPU ownership, exact peak or sustained VRAM qualification has been performed. Most of the decoder and all Talker/CodePredictor operations still run on CPU in this diagnostic. No production dispatch or service state changed.
