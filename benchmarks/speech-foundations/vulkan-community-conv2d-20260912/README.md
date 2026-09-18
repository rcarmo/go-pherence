# Community-1 Vulkan CHW 2-D convolution

`VkConv2DCHWF32` adds the model-free spatial convolution primitive needed by the Community-1 WeSpeaker ResNet34 path. It accepts contiguous F32 `X[inChannels,frequency,frames]`, `W[outChannels,inChannels,kernel,kernel]` and `Out[outChannels,outFrequency,outFrames]`. It is explicit opt-in; CPU, model and service defaults are unchanged.

## Contract

The checked envelope is deliberately limited to the architecture's bias-free `3×3/pad1` and `1×1/pad0` convolutions, each with stride 1 or 2, 1–256 input/output channels, frequency 1–80 and frames 1–4096. Input and output tensors stay within the existing four-million-element Community-1 block bound. Groups, dilation, bias, activation, BatchNorm and checkpoint policy are excluded.

The shader computes one 16-position × 16-output-channel tile per workgroup. Two 256-F32 shared arrays hold implicit-im2col input and weight tiles. Reduction order is ascending `inputChannel,kernelFrequency,kernelTime`, matching the model's flattened weight order. Output must be disjoint from input and weights; the two read-only buffers may share storage. Shape, byte extent, grid and storage-buffer limits are checked before command recording. There is no hidden fallback or host content scan.

## Offline verification

- The source schedule model covers eight geometries, both kernel sizes, both strides, odd/tile-boundary dimensions, four shuffled group/lane schedules and 13,936 output comparisons. Maximum absolute error against an independently structured float64 direct convolution is `1.3716412015085666e-06`, within the fixed `2e-5 + 2e-5*abs(reference)` diagnostic envelope.
- A non-symmetric 3×3 analytic fixture checks tap orientation and zero padding exactly.
- Focused tests cover descriptor ranges, nine-word push ABI, grid geometry, copied stages, nil/rank/shape/storage/kernel/stride/padding rejection, exact/partial output alias rejection, cancellation, device limits, constructor/close and plan retention through drain.
- All 19 embedded shader contracts pass the closed parser. The new shader has local size `16×16×1`, 2048 shared bytes, three storage descriptors and 36 push bytes.
- The static checker passes six tests with 61 assertions. Stored and embedded SPIR-V SHA-256 is `b5de79ccaca0ba3de8f13416f97fbb144ac1f398839ecddc63a12230a4ec6ba6`.

No Vulkan device, trained checkpoint, audio, service, deployment, push or performance run was used. Static validation and a Go schedule model do not establish native driver arithmetic, performance or trained graph quality. Resident Community-1 graph ownership, coefficient preparation, stage composition, device-loss recovery and CPU/full-Vulkan/hybrid whole-job comparison remain open.
