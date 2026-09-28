# Qwen3-TTS three-convolution GPU decoder timing

On one released 64-frame input, five warm **decoder-only** samples put the existing CPU path at **26.706 s/op median** and the diagnostic three-convolution GPU path at **26.766 s/op median**. The 0.22% difference by medians is too small to establish a useful speed difference from this block-ordered sample. Production synthesis remains CPU-only.

## Measured boundary

The benchmark loads the approved Qwen3-TTS 0.6B CustomVoice speech-tokenizer checkpoint and the hash-pinned 64-frame codes and independent Rust/Candle waveform outside timing. It runs the existing `decodeCodes` on CPU and a test-only `decodeCodesWithDiagnosticConvs` path that substitutes the prepared NVIDIA pre-convolution, decoder-initial convolution and final convolution. The remainder of the decoder runs on CPU. Warm GPU setup packs/uploads weights and allocates request-local buffers before timing. Both decoded waveforms pass the unchanged **1.6e-6** reference gate: CPU maximum error **1.28487591e-6**, GPU hybrid **1.28522515e-6**. The hybrid differs from the CPU waveform by at most **5.96046448e-8**.

Timed GPU calls include host im2col, activation uploads, 1,937 SGEMM launches, partial downloads, CPU reductions/layout conversion and the remaining CPU decoder. They exclude checkpoint load, kernel/context initialization, GPU buffer setup, correctness checks and full Talker/CodePredictor inference. One observed GPU buffer/weight setup took **61.16 ms**, excluding model and kernel load.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR=/workspace/tmp/qwen3tts-cap64-20260926/reference1 \
  go test ./model/qwen3tts -run '^$' \
  -bench '^BenchmarkDecoderThreeGPUConvsProbe$' \
  -benchmem -benchtime=1x -count=5 -v -timeout=1200s
```

| Decoder-only path | Median time, five samples | Median B/op | Median allocs/op |
|---|---:|---:|---:|
| Existing CPU | 26,706,439,409 ns | 1,125,656,832 | 273 |
| Prepared three-stage GPU hybrid | 26,765,810,451 ns | 1,121,898,248 | 48,696 |

The benchmark runs five CPU calls followed by five GPU calls on one host (`i7-12700`, `GOMAXPROCS=6`, RTX 3060). Each timed iteration runs a full decoder decode, but the input codes and model remain fixed; there is no randomized order, uncertainty interval or concurrent load control. The much larger GPU-path allocation count includes repeated host-side work in the diagnostic callbacks. Raw timings and checks are under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/threeconv-bench-1x5.log`.

The [live three-convolution correctness check](qwen3-tts-gpu-threeconv-20260927.md) records the distinct upstream Rust boundary drift and sampled GPU memory. This timing run establishes neither a full TTS request speedup nor production GPU admission; remaining decoder stages, Talker/CodePredictor, cancellation, concurrent ownership, peak/retained VRAM and full-request latency remain unqualified.
