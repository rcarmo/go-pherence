# Qwen3-TTS 64-frame pre-convolution: CPU versus host-staged GPU

The numerically exact blocked GPU diagnostic is slower and allocates more than the production CPU pre-convolution at the released 64-frame shape on this host. It is not a candidate for production dispatch in its present host-staged form.

## Workload

- Baseline Go revision `b83e8038e76318fd795ce1671476b8f3fef96fc8`, clean before this benchmark; Go 1.26.3, linux/amd64, Intel i7-12700, `GOMAXPROCS=6`. NVIDIA GeForce RTX 3060, driver 610.57.04, idle VRAM 33 MiB.
- Approved pinned `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision `85e237c12c027371202489a0ec509ded67b5e4b5`; the opt-in benchmark hashes decoder config/weights and the 64-frame `quantized.full.f32le` input and `preconv.full.f32le` independent Rust output before use. A warm-up verifies both CPU and GPU outputs against all 65,536 independent F32 values bitwise.
- Unit of work: one released pre-convolution at input `[512,64]`, weight `[1024,512,3]`, output `[1024,64]`. The CPU sub-benchmark calls the existing `decoderConv1D.forward` including its per-call scratch and output. The GPU sub-benchmark includes per-call causal im2col, weight transpose, three 512-wide host SGEMMs with uploads/downloads, F32 block accumulation, bias and channel-major output assembly. Both exclude checkpoint loading; GPU kernel preparation was warmed first. GPU time includes transfer and host preparation. This comparison does not model a future resident-weights/activations design.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  go test ./model/qwen3tts -run '^$' -bench '^BenchmarkDecoderPreConvReleased$' \
  -benchmem -benchtime=30x -count=10 -timeout=300s
```

Raw ten-sample output is in `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/preconv-bench-final-30x10.log`. Median CPU: **2,238,932 ns/op**, 1,277,955 B/op (sample median), 4 allocs/op. Median host-staged GPU: **9,854,320 ns/op**, about 8,392,794 B/op (sample median), 233 allocs/op. The GPU path took **4.40×** the CPU time for this single stage. Samples ran as a CPU block followed by a GPU block; they were not interleaved. The ratio is a diagnostic on one machine and one shape, not a latency claim for a device-resident decoder. Full synthesis throughput, peak VRAM and retained device memory were not measured.

The [numerical pre-convolution gate](qwen3-tts-gpu-preconv-20260927.md) also records the unblocked (`4.2915e-6`) and compensated (`2.3842e-6`) SGEMM failures against the unchanged `1e-6` diagnostic threshold. The exact blocked probe uses three launches and 6,684,672 host-to-device plus 786,432 device-to-host bytes per call. Improving this stage requires a separately profiled device-resident design that preserves the three-block F32 reduction and the CPU waveform gate; the benchmark alone does not justify that work or production promotion.
