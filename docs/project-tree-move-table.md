# Project tree move table

Applied by `scripts/mass_move_project_tree.py`.

## Vulkan

- Runtime/device files moved to `backends/vulkan/runtime`.
- Buffer files moved to `backends/vulkan/memory`.
- Dispatch/kernel creation files moved to `backends/vulkan/dispatch`.
- GLSL/SPIR-V files moved to `backends/vulkan/shaders`.
- Operation wrappers and parity tests moved to `backends/vulkan/ops`.

## NVIDIA

- Device buffers moved to `backends/nvidia/memory`.
- Streams/stats moved to `backends/nvidia/streams`.
- Compiler/module loading moved to `backends/nvidia/modules`.
- Launch sizing moved to `backends/nvidia/launch`.
- Operation families moved to `backends/nvidia/ops/{bf16,q4,mlx,nvfp4,lmhead,attention,matmul}`.
- Expert pool logic moved to `backends/nvidia/experts`.

## Model

- Speculative/MTP files moved to `model/speculative`.
