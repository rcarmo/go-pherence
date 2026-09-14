# Community-1 four-output SincNet FMA — 13 September 2026

A four-output Plan 9 AVX2/FMA kernel reduced the pinned ten-second SincNet call from 339.170ms to 106.482ms. With the packed Vulkan LSTM and 32×32 Vulkan convolution tile, the complete 30-second Community-1 hybrid median is now 13.280 seconds, `2.073×` faster than the original Vulkan median.

## Change

`FMAColumns4F32Checked` computes four output rows from one packed input tile. Each AVX2 lane spans an independent output frame, and four vector accumulators span four output channels. Every accumulator starts at positive zero and visits K in ascending order, matching the existing scalar and one-output kernels bit-for-bit. The input vector is loaded once for four output channels.

The SincNet `SincNetSIMDFMA` path uses this kernel in groups of four channels. The existing one-output checked kernel handles any remainder. Packing, bias addition, pooling, normalisation and activation semantics are unchanged. Unsupported processors use the exact scalar implementation. No mode, model, tolerance, service or default changed.

## Measurements

Environment:

- model revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`;
- Intel i5-1340P host, `GOMAXPROCS=2`, `CGO_ENABLED=0`;
- full runs use `Intel(R) Iris(R) Xe Graphics (RPL-P)` through the pinned Intel ICD;
- pinned segmentation, embedding and PLDA assets plus the 480,000-sample public WAV;
- each trained process bounded to 180 seconds.

A saved pinned ten-second PCM trace measured twelve SincNet calls:

```text
106.482052ms average; 105.833–107.690ms range
```

The pre-change stage diagnostic measured 339.170139ms per window. The isolated stage improvement is `3.185×`.

Three complete trained runs measured:

| Run | 21-window run |
|---|---:|
| 1 | 13.288150s |
| 2 | 13.273328s |
| 3 | 13.279526s |

The median is 13.279526 seconds. It is `1.366×` faster than the preceding 18.138189-second convolution-tile median and `2.073×` faster than the original 27.531066-second Vulkan median. Against the retained 49.235-second existing-CPU median it is `3.708×`; those CPU and Vulkan runs were not balanced in one alternating trial.

Each trained run completed all 21 windows with 37 clustering rows, two clusters, 13 full turns, 12 exclusive turns and 84 ambiguous frames. The result stayed byte-identical to the retained baseline:

```text
111e98ad0a6d1fb5dc61bcf6db4ff0d3d44485cc66c09efcabf7736b7d8f4ee3
```

Tracked Vulkan memory remained 87,219,248 bytes in two allocations and returned exactly after close. Existing DER scores and the `qualified=false` decision stay unchanged. This one-sample diagnostic uses explicit `LowestIndexTies`; the default policy rejects ambiguity.

## Verification

- exact scalar/assembly output across column tails 1–32 and K lengths through 400;
- finite-input, shape and alias rejection before writes;
- zero allocation for the checked kernel;
- Linux guard pages at both ends for columns 1–65 and K lengths 1, 3 and 7;
- MXCSR round-mode, DAZ and FTZ rejection with untouched output;
- native and forced AVX2/FMA-off SincNet endpoint and cancellation tests, ten repeats each;
- affected SIMD, Community, speech-job and server package tests;
- Linux ARM64 compilation for SIMD and Community packages;
- assembly object inspection confirms no calls in the new hot kernel;
- package vet reports only the three pre-existing `q8dot_amd64.s` return-offset warnings;
- three complete trained runs pass with byte-identical output and exact Vulkan cleanup.

Temporary timing source was removed after measurement. No service ran or changed. No model asset, corpus, deployment setting, tolerance or default changed.
