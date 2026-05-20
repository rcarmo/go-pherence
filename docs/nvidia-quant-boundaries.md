# NVIDIA quantized runtime boundaries

This page records deliberate NVIDIA runtime support boundaries for Q4/GPTQ and NVFP4 so coverage tables can distinguish implemented paths from intentionally deferred work.

## Q4 / GPTQ

Implemented:

- Symmetric GPTQ/Q4 GPU GEMV and batch-1/mini-batch GEMM paths are owned by `backends/nvidia/runtime` with PTX under `backends/nvidia/ptx/q4`.
- Upload and dispatch validate dimensions, packed qweight sizes, scale layout, group indices, buffer byte sizes, and product overflow before allocation or launch.
- CPU fallback logic imports the owning scalar package (`backends/simd/quant/q4`) rather than `runtime/quant`.

Boundary:

- Asymmetric GPTQ `qzeros` is supported in CPU scalar dequant/reference code, but there is no NVIDIA asymmetric `qzeros` dispatch path today.
- Add a NVIDIA asymmetric qzeros smoke only if a model-loading path intentionally routes asymmetric GPTQ to GPU. Until then, the NVIDIA quantized runtime is symmetric-Q4 only and asymmetric paths should remain CPU/reference fallback.

## NVFP4

Implemented:

- `GPUNVFP4Weight` uploads packed FP4 bytes and F8 scale bytes as raw buffers.
- NVIDIA dequant-to-F32 fallback materializes dense F32 weights for correctness-first validation.
- Dense GEMV can consume the materialized F32 representation.
- `NVFP4KernelSpec` defines the packed/native GEMV/GEMM interface contract: row-major packed weights, F8 scales, F32 inputs/outputs, batch semantics, u32 NVIDIA-interface limits, group size validation, and overflow guards.
- Native NVFP4 tensor-core dispatch is hardware-gated and disabled until real kernels and hardware smoke tests exist.

Boundary:

- Public NVFP4 generation remains gated. Synthetic dequant/GEMV parity is necessary but not sufficient.
- Packed/native NVFP4 GEMV/GEMM is interface-only today; it should be treated as pending, not silently equivalent to the dense F32 fallback.
- NVFP4 LM-head support is future work unless a real checkpoint quantizes the LM head. Inspected NVIDIA Qwen3 checkpoints keep embeddings and LM head BF16.
