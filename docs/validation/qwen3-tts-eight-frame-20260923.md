# Qwen3-TTS capped greedy CPU: eight-frame validation

The native Go 0.6B CustomVoice CPU reference matches the pinned Rust/Candle eight-frame Ryan/English `Hello world` oracle: semantic IDs `[1995, 215, 212, 1181, 462, 251, 530, 122]`, all 120 acoustic IDs, and a 15,360-sample waveform with maximum absolute difference `9.164214134216309e-7` (limit `1e-6`). This is a 640 ms, cap-limited greedy probe. The released prompt never sampled codec EOS; a synthetic selector test checks early EOS, partial-frame exclusion, output ownership and a one-frame cap. The fixed 2–4 frame APIs still require their exact frame counts. No stochastic sampler, naturally observed EOS stop, intelligibility or production-rate synthesis has been qualified.

The source is `model/qwen3tts/two_frame_cpu.go`. Fixtures are under `model/qwen3tts/testdata/customvoice_0b6_ryan_hello/`; `reference.json` pins SHA-256 for the model, tokenizer, independent script, frame-major codes and waveform. Rust/Candle is `TrevorS/qwen3-tts-rs@711ceee07cad92673f86de8997bdf54c30caa49f`; official model revision is `85e237c12c027371202489a0ec509ded67b5e4b5`. Copy `scripts/qwen3tts_oracle_eight_frames.rs` to that checkout's `examples/oracle_eight_frames.rs`, then run `cargo run --release --example oracle_eight_frames --no-default-features --features cpu -- <pinned-model-dir> <output-dir>`. The script asserts all eight tokens are non-EOS. Its regenerated code and waveform SHA-256 values matched the committed files. Weights stay outside git.

## Reproduce correctness

```sh
export GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=<pinned-model-dir>
GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^TestCappedGreedyCPUReleasedEightFrames$' -count=2 -v
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./model/qwen3tts -run '^TestCappedGreedyCPUReleasedEightFrames$' -count=1 -v
GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run 'TestCappedGreedyCPU|TestDecoderConvIndexedMatchesScalarReduction' -count=10
```

The opt-in released gate passed twice (31.16 and 30.53 s), then under race instrumentation (225.33 s). The eight-frame waveform error was identical on each run. Existing 2–4 frame released gates passed after the decoder change without widened tolerances. A combined released race run timed out at 340 s **after** eight-frame and first-frame decoder tests passed but during the four-frame test; its unfinished four-frame result is not a pass. The isolated eight-frame race run completed. The model-free whole-tree CPU race suite, vet, build, docs/layout checks and Linux arm64/riscv64 package and probe cross-builds passed. Foreign cross-builds were not executed. Package-wide ordinary-test statement coverage was 77.9%; this does not meet the repository's 90% changed-package target and the released gate was not in that coverage run.

## Profiling and one bounded change

The baseline was Git `014019a0` plus the uncommitted capped generator and benchmark, before the decoder convolution edit. Go `1.26.3` on Linux/amd64, Intel i7-12700, default effective 6 CPUs, native SIMD enabled, CUDA disabled. Each benchmark operation generates eight frames from an already loaded model and constructed prompt; the model hash is checked before loading, outside the timer. It includes Talker prefill, all 15 codes per frame and full Decoder12Hz waveform generation; it excludes checkpoint I/O. Run: `GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^$' -bench '^BenchmarkCappedGreedyCPUReleasedEightFrames$' -benchtime=1x -count=5 -benchmem` with the same environment variable. Before/after were sampled sequentially on the same host, not interleaved; load or thermal drift can affect the comparison.

| State | Five seconds/op samples | Mean | Bytes/op | Allocs/op |
|---|---|---:|---:|---:|
| Original scalar indexing | 32.062, 33.531, 35.201, 33.907, 34.560 | 33.91 s | 265,818,112 | 18,245 |
| Pre-sliced channels/taps, same F32 reduction | 28.460, 29.419, 28.930, 28.102, 30.072 | 28.93 s | 265,818,112 | 18,245 |

`benchstat` (`golang.org/x/perf@v0.0.0-20260709024250-82a0b07e230d`) gives -14.68% sec/op (p=0.008, n=5); five samples are too few for its 95% confidence interval. The indexed convolution is bitwise identical to the old scalar reduction for deterministic dense/depthwise, dilation/padding and short/tail shapes. The eight-frame released waveform difference stayed `9.164214134216309e-7`; the 2–4 frame limits are unchanged.

A separate baseline CPU profile sampled `decoderConv1D.forward` at 150.51/197.16 s flat (76.34%), `decoderTransConv1D.forward` 15.60 s (7.91%), and Decoder12Hz at 169.32 s cumulative (85.88%). After indexing, `decoderConv1D.forward` still consumed 63.05/86.41 s flat (72.97%), with transposed convolution at 7.28 s (8.42%). The pprof runs include setup outside benchmark timing, so their totals are attribution, not speed evidence. Sampled heap profiles were dominated by one-time safetensors/model loading and tokenizer parsing: they must not be read as warm generation allocation breakdown. Warm generation stayed at 265.8 MB/18,245 allocations per request. Cold-load latency, peak RSS, retained heap after reuse, exhaustive allocation tracing and a timed end-to-end load-plus-synthesis benchmark have not been measured in this slice.

CodePredictor still builds a per-frame cache. Its warm workspace is an open target, but the current CPU profile does not justify prioritising it above Decoder12Hz; further changes need new profiling and the same independent numerical gates. NVIDIA, streaming, arbitrary-length sampling, 1.7B and speech-quality qualification remain separate work.
