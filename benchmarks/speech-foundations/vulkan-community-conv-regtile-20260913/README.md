# Community-1 Vulkan convolution register tile — 13 September 2026

A 32×32 four-output register tile reduced the pinned 30-second Community-1 hybrid median from 21.624 seconds to 18.138 seconds after the packed-LSTM change. Combined speedup over the original 27.531-second Vulkan median is `1.518×`.

## Change

The existing `VkConv2DCHWF32` workgroup remains 16×16 invocations. Each invocation now owns four outputs in a 32-position by 32-output-channel tile. Cooperative input and weight tiles grow from 16×16 to 32×32, and the K tile grows from 16 to 32. The implicit-im2col indexing, CHW layout, row-major weights, padding, strides and ascending-K accumulation order remain unchanged.

The wrapper dispatch grid changes from `ceil(outChannels/16) × ceil(outSpatial/16)` to `ceil(outChannels/32) × ceil(outSpatial/32)`. The shader uses 8,192 shared bytes, within the existing Vulkan admission checks. No API, model, tolerance, service or default changed.

## Native results

Environment:

- model revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`;
- device `Intel(R) Iris(R) Xe Graphics (RPL-P)`;
- `/usr/share/vulkan/icd.d/intel_icd.x86_64.json` selected through both Vulkan ICD variables;
- `GO_PHERENCE_DISABLE_NVIDIA=1`, `GOMAXPROCS=2`, `CGO_ENABLED=0`;
- pinned segmentation, embedding and PLDA assets plus the 480,000-sample public WAV;
- physical-device runs bounded to 180 seconds.

Synthetic native block, trunk and hybrid embedding comparisons pass. Their maximum absolute errors are `7.45e-9`, `3.73e-9` and `9.31e-10` respectively.

A saved trained five-second Fbank trace isolates the complete resident trunk. Twelve calls took 1.187533 seconds, or 98.961ms per call including the first 118.552ms call. The eleven later calls are 96.151–98.554ms. This timing establishes the current kernel cost; no equivalent pre-change isolated trunk run was recorded.

Three complete trained runs measured:

| Run | Construct | 21-window run | Close |
|---|---:|---:|---:|
| 1 | 38.909ms | 18.184040s | 1.097ms |
| 2 | 35.928ms | 18.138189s | 0.977ms |
| 3 | 38.366ms | 18.095650s | 0.637ms |

The median is 18.138189 seconds. The packed-LSTM checkpoint median was 21.623503 seconds, so the convolution tile adds `1.192×` and reduces latency by 16.12%. The original Vulkan median was 27.531066 seconds; the combined speedup is `1.518×` and the combined latency reduction is 34.12%.

Each run completed all 21 windows with 37 clustering rows, two clusters, 13 full turns, 12 exclusive turns and 84 ambiguous frames. Tracked Vulkan memory stayed at 87,219,248 bytes in two allocations and returned to the initial state after close. The serialised result stayed byte-identical to the retained baseline:

```text
111e98ad0a6d1fb5dc61bcf6db4ff0d3d44485cc66c09efcabf7736b7d8f4ee3
```

The one-sample DER values and `qualified=false` decision do not change. This diagnostic uses explicit `LowestIndexTies`; the default policy still rejects ambiguity.

## Verification

- eight offline convolution geometries cover odd tails, 1×1 and 3×3 kernels, stride one and two, padding and shuffled lane/group schedules;
- ten repeated focused Vulkan and Community owner checks pass;
- all 23 shaders pass `spirv-val`, stored/embedded identity, rebuild validation and normalised rebuild comparison;
- synthetic native block, trunk and hybrid embedding parity pass;
- three complete trained runs pass with byte-identical output and exact post-close memory return;
- affected package tests, `go vet`, Linux ARM64 test compilation and `git diff --check` are run before commit.

Temporary timing source was removed after measurement. No service ran or changed. No model asset, corpus, deployment setting, tolerance or default changed.
