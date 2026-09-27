# Pocket TTS native ARM64 preset-voice check

The released preset-voice Pocket TTS path passes correctness and zero-warm-allocation checks on a CIX P1 ARM64 board, but its measured generation is slower than real time there. This run did not test raw-audio voice cloning, training or native RVV.

The board ran Linux/ARM64 with `GOMAXPROCS=2`, `nice -n 10` and NVIDIA disabled. The test binary was built from go-pherence `f0259ae1` with Go 1.26.2 and `CGO_ENABLED=0`; binary SHA-256 was `7686e3ac503105eeb82e8a73e18bddf5681410bdb144f44837554da6efb018e0`. The imported preset-only English model, tokenizer and Alba voice assets matched the pinned SHA-256 values in `model/pockettts/released_assets_test.go`: `916ccd26…52344f`, `f498428e…d8c6687` and `69c32db6…1d8845`. No model assets entered Git. The first test attempt lacked the package's relative `testdata/` files; it failed before inference and was rerun after transferring a SHA-checked fixture bundle.

The selected released checks passed: FlowLM prefill (reported max absolute difference 2.7194619178771973e-6, EOS difference 6.67572021484375e-6), one-frame end-to-end generation, one-frame Mimi parity, batch versus frame streaming, tokenizer parity, tensor inventory and ten warm five-frame calls with zero heap allocations.

Three fresh-process benchmark runs per case, five warm calls per process, reported:

| Frames | Audio duration | Median ARM64 time | Range | Heap per call |
|---:|---:|---:|---:|---:|
| 1 | 80 ms | 502,182,248 ns | 502,129,729–529,766,794 ns | 0 B, 0 allocs |
| 5 | 400 ms | 1,861,029,346 ns | 1,792,832,368–1,873,433,705 ns | 0 B, 0 allocs |
| 25 | 2 s | 5,072,630,502 ns | 5,070,200,702–5,081,036,158 ns | 0 B, 0 allocs |

These are board measurements under its current workloads. The prior i7-12700 [inference record](pocket-tts-native-inference-2026-09-22.md) reported 51–54 ms, 143–147 ms and 398–427 ms for the same frame counts with different hardware and worker settings. The ARM64 25-frame median is about 2.54 times the generated audio duration; it does not pass a real-time target. Pinned outputs and allocation results establish a bounded native correctness path, not latency or production readiness.

A separate three-call ARM64 CPU profile measured 5,239,560,859 ns/op and sampled 16.19 CPU seconds, including setup. `FMA32Scalar` was 7.57 seconds flat (46.76%), `BF16DotF32` 2.99 seconds flat (18.47%) and `fmaMatrixScalar` 2.00 seconds flat (12.35%). `MimiDecoderCPU.DecodeFrameInto` was 11.54 seconds cumulative (71.28%), with `CausalConv1D.ForwardInto` at 10.16 seconds cumulative (62.75%). Percentages are CPU-sample shares, not wall-time shares. For causal convolutions with widths up to 256, `FMAMatrixF32Checked` selects `fmaMatrixScalar` on ARM64 to preserve the specified ascending-K F32 FMA order; BF16 x4 dot uses a scalar fallback outside amd64. A faster kernel needs numerical and released-waveform parity before selection. No kernel or default was changed in this measurement.

Raw logs, profile and transferred SHA manifest remain under `/home/agent/pockettts-native-20260927/` on the board.
