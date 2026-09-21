# OmniVoice Vulkan integration notes

This package does not currently implement an OmniVoice Vulkan backend.

## What exists already

`backends/vulkan` exposes low-level compute kernels and runtime wiring for:

- `vec_add_f32`
- `vec_add_bf16`
- `rms_norm_f32`
- `rms_norm_bf16`
- `rms_norm_no_scale_f32`
- `gemv_f32`
- `gemv_bf16_mixed`
- `silu_mul_f32`
- `gelu_tanh_mul_f32`
- `rope_partial_f32`
- `attention_score_f32`

The SpacemiT board wrapper in `backends/spacemit/board/vulkan.go` shows the current usage model: allocate Vulkan buffers, upload host slices, dispatch one kernel, then download results. It explicitly falls back to CPU/SIMD unless `GO_PHERENCE_VULKAN_HOST_ROUNDTRIP=1` is set, because per-op host round trips are slower than the CPU path.

## Why OmniVoice does not claim Vulkan support yet

`model/omnivoice/block.go` is a CPU/SIMD implementation. It calls checked SIMD APIs directly for:

- linear projections via `SgemmNTTo` and `SgemmNNTo`
- RMSNorm
- softmax
- SiLU-gated MLP

It also performs Go-side orchestration for:

- Q/K/V head packing
- grouped-query head mapping
- RoPE table preparation and per-head dispatch (rotation arithmetic uses vector kernels)

Mask and residual additions now use vector kernels. Softmax and SiLU exponentials still use scalar `math.Exp` on amd64; capability reporting therefore keeps `full_graph_simd=false`.

A truthful Vulkan backend would need more than isolated kernels:

1. persistent GPU residency for q/k/v/o and MLP weights
2. a reusable scratch plan for packed heads, scores, and attended outputs
3. projection coverage for the actual matrix shapes used by the block
4. an end-to-end attention path, including softmax and score×value accumulation
5. parity-tested dispatch from the OmniVoice block itself
6. explicit policy for hardware Vulkan vs software drivers such as llvmpipe

## Practical caveats

- `VkGemvF32` is not enough to replace the current projections, which are matrix-matrix operations over token batches.
- `attention_score_f32` only covers QK score generation. The block still needs masking, softmax, and score×V.
- The current Vulkan buffer model is host-visible mapped memory. Without a residency plan, explicit Vulkan mode would mostly benchmark transfer overhead.
- Software Vulkan (`llvmpipe`, `lavapipe`, `swiftshader`) is not a useful inference target for OmniVoice.

## Feasible next step

The smallest honest Vulkan milestone would be a separate experimental path that:

- uploads one layer's weights once
- reuses device buffers for activations and scratch across repeated block calls
- implements the full attention subgraph for parity tests
- keeps explicit `implemented=false` in public capability discovery until end-to-end block execution is wired and validated
