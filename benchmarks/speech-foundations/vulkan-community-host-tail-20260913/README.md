# Community-1 Vulkan host-tail placement decision — 13 September 2026

## Decision

Keep mask-dependent statistics pooling and the final embedding projection on CPU. Moving them to Vulkan is not justified by measured cost and cannot materially improve the current 27.5-second trained hybrid run.

The current `VulkanEmbedding` downloads the fixed-window ResNet trunk output and then runs:

1. legacy-nearest mask resizing and weighted mean/sample-standard-deviation pooling;
2. one 256×5120 projection for each admitted local-speaker mask.

This preserves dynamic mask semantics and uses the existing checked Plan 9 `GemvRows` path. The resident Vulkan trunk remains the dominant neural placement.

## Measurement

A temporary test used the pinned `public-6-11s.safetensors` trained embedding trace:

- model revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`;
- 5 seconds / 498 Fbank frames / 63 CNN frames;
- trunk output `[256,10,63]` (161,280 F32 values);
- three masks and 256-dimensional outputs;
- `GOMAXPROCS=2`, three process-level repeats, 20 calls per measured arm.

| Repeat | WeSpeaker Fbank / call | CPU pool + 3 projections / call |
|---:|---:|---:|
| 1 | 14.069 ms | 1.339 ms |
| 2 | 13.965 ms | 1.374 ms |
| 3 | 13.840 ms | 1.361 ms |

The trained 30-second hybrid run uses 21 overlapping 10-second windows and took 27.53 seconds. Conservatively doubling the measured 5-second pool cost gives about 2.75 ms/window, or about 58 ms for all 21 windows: roughly 0.21% of total runtime. Even eliminating that cost entirely cannot approach the required 1.5× speaker-stage improvement. Fbank is larger but remains about 0.59 seconds under the same deliberately conservative linear estimate and is not part of a GPU pool/projection implementation.

The output transfer is approximately 1.28 MB per 10-second window (`256×10×125×4`), about 26.9 MB across 21 windows. Keeping pooling on-device would avoid that transfer but would require dynamic mask upload, weighted reductions, square root, support metadata, a resident 5.24 MB projection, new ownership/error paths and a later embedding download. No measured benefit supports that complexity.

## Scope and limits

This is a warm component measurement using saved trained tensors, not a new whole-job GPU run, energy result or device-transfer benchmark. The 10-second estimate is intentionally conservative and not reported as measured latency. It is sufficient for a YAGNI placement decision because the host tail is two orders of magnitude too small to close the whole-job target.

No runtime code, model, tolerance, service, deployment or default changed. The temporary timing test was removed after the three repeats. Future work should profile the remaining resident CNN/LSTM, CPU SincNet/Fbank and orchestration stages rather than add Vulkan pooling/projection.

## Raw output

See `host-tail-timing.txt`.
