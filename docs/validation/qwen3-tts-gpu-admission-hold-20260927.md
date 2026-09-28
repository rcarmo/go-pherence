# Qwen3-TTS GPU admission hold after decoder profiling

Keep Qwen3-TTS synthesis CPU-only. Three GPU convolutions in a live 64-frame decoder pass the independent waveform gate, but their [decoder-only timing](qwen3-tts-gpu-threeconv-decoder-benchmark-20260927.md) is effectively tied with CPU on one fixed input and uses many more Go allocations. The remaining decoder and Talker/CodePredictor run on CPU. There is no full-request GPU latency, concurrency, cancellation or exact peak/retained VRAM qualification.

## Current decoder bottleneck

An isolated, opt-in CPU-only benchmark loaded the approved CustomVoice decoder, hash-checked the 64-frame codes and independent Rust waveform, ran an untimed correctness decode, then timed one warm `decodeCodes` call. Both decoded waveforms matched the unchanged `1.6e-6` gate with maximum error **1.28487591e-6**. One observed timed call took **27.315 s**; it is not a latency distribution. The Go CPU profile sampled the **whole process**, including model loading and the untimed correctness decode. Across 56.07 sampled CPU seconds, `decoderProjectionRows` accounted for **34.79 s flat (62.05%)**, `decoderConv1D.forwardCandleOrder` for **5.08 s flat (9.06%; 10.07 s cumulative)**, `math.sin` for **4.87 s (8.69%)**, `gebpMicroKernel` for **4.05 s (7.22%)**, and `SgemmNN` for **2.37 s (4.23%)**. These CPU-time percentages are not fractions of the 27.315 s wall-time call. Profiling with the three-convolution benchmark also found `decoderProjectionRows` at 69.91/112.77 sampled CPU seconds (61.99%), but that mixed profile includes CPU/GPU setup and two untimed correctness decodes.

```sh
GOMAXPROCS=6 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR=/workspace/tmp/qwen3tts-cap64-20260926/reference1 \
  go test ./model/qwen3tts -run '^$' \
  -bench '^BenchmarkDecoderCPUOnlyPinnedProfile$' -benchtime=1x -count=1 \
  -cpuprofile=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/decoder-isolated-cpu-profile.pprof \
  -timeout=300s
```

The isolated profile and log are under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/decoder-isolated-cpu-profile.*`. `decoderProjectionRows` preserves the Candle-order blocked F32 reduction via `math.FMA`; replacing its arithmetic without independent stage and waveform gates could lose parity. A GPU projection trial would also need a batched layout and transfer plan: one-row host-staged operations are not justified by this CPU profile. The published projection diagnostics cover individual pinned stages, not all decoder projections or their live chained outputs.

## Decision and reopening criteria

No production GPU dispatch or GPU memory-resident model path is admitted. A next bounded experiment should isolate batched decoder projections on pinned Rust inputs, compare live CPU boundaries under explicit numerical gates and measure full decoder latency **including** packing/transfers against the same fixed CPU request. This experiment does not authorise promotion by itself. Before promotion, qualify the remaining decoder, Talker and CodePredictor with independent codes/waveforms and the unchanged waveform gate; measure full-request latency and allocations on repeated inputs; test concurrent owned requests, cancellation and failure recovery; and measure actual in-flight peak plus retained VRAM under a stated budget. The 100 ms GPU-wide polling maxima from earlier diagnostics are lower-bound observations, not peak admission evidence.
