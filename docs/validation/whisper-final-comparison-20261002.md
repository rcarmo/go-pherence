# Final whisper.cpp vs Go comparison — 2 October 2026

Native Go Whisper large-v3-turbo Q5_0, at commit `d16da7bb`, is faster than pinned whisper.cpp on Sigma (i5-1340P, Intel Iris Xe) in every matched configuration. Both engines use Vulkan, flash attention and Silero VAD, run in the same window, and the order alternated per fixture.

## Method

- **whisper.cpp:** the `refresh-original-final` reusable-context harness with Vulkan, `flash_attn=1`, 4 CPU threads, greedy decoding and Silero v6.2.0 VAD at default parameters. It is linked against the same pinned libraries as the [workflow refresh](whisper-workflow-refresh-20261001.md), and its word timing was off.
- **Go:** backend `vulkan-original-q5-padded-integer-dot-mmq-tanh` with original decoder/window compatibility, GPU cross K/V and the packed-Q5 CPU decoder. Compact native Silero VAD, word timing off.
- **Guards:** both run under CPU4/8GiB/no-swap, Intel GPU, and the Qwen-idle guard.
- **Order:** ABBA. Pass A ran the original first for each fixture; pass B ran Go first. Each arm made 5 requests in one process.
- **Statistic:** medians over all 10 requests per engine, including each process's first request.

## Results

| Fixture | whisper.cpp | Go | Δ | Pass A / B (orig → Go) | Output |
|---|---:|---:|---:|---|---|
| JFK (11 s) | 3.019 | **2.786** | −7.7% | 3.019→2.786 / 3.013→2.770 | segments, text and times identical |
| PT row 0 | 3.012 | **2.779** | −7.7% | 3.012→2.779 / 2.996→2.774 | identical |
| PT2 (44 s, 2 windows) | 6.212 | **6.034** | −2.9% | 6.212→6.034 / 6.205→6.003 | identical (6 segments) |
| JFK + VAD | 3.041 | **2.859** | −6.0% | 3.041→2.859 / 3.025→2.850 | same text; boundaries 7.46/8.21 vs 7.47/8.19 s |
| groups + VAD (54 s, compact) | 3.285 | **3.195** | −2.7% | 3.273→3.173 / 3.285→3.195 | same words; Go 4 segments, original 2 |

Every request within an arm gave identical output. The original's first request in each process takes about 4.4 s and Go's about 2.8 s; excluding first requests, Go still wins on every fixture.

## What changed since the 1 October refresh (JFK 6.31 s → 2.79 s)

| Change | Commit |
|---|---|
| GPU cross K/V and fused AVX2 Q5_0 CPU decoder | c21fb417 |
| Parallel cross heads and sparse mel | ef184c91 |
| uvec4 Q8_1 MMQ loads | 744007bf |
| [48-query flash attention](whisper-attention-q48-20261002.md): 54.2 → 33.1 ms per layer; whisper.cpp FA 49.4 | 3a4b1f7e |
| [Batched prompt prefill](whisper-interleaved-comparison-20261002.md) | 6dd35fbc |
| [Fewer-uop Q5_0 dot](whisper-q5-cpu-kernel-20261002.md) | d16da7bb |

Each change was bit-identical, with 10 fixtures × 5 requests giving output identical to the previous arm.

## Differences that remain (not speed)

- **Groups VAD segmentation:** both engines produce the same words, but Go splits each speech group into two segments.
- **Word timing:** not compared. whisper.cpp with flash attention uses its token-timestamp heuristic (DTW is disabled). Go uses cross-attention DTW in an extra pass and fails closed on words that cross a removed VAD gap.
- **Decoder placement:** the Go decoder runs on the CPU (packed Q5, AVX2), while the original's runs on the GPU. The encoder, attention and cross K/V run on Vulkan.
- **Scope:** model load and cold setup are outside these request timings. Results cover Sigma only.
