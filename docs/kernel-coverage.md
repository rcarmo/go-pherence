# Kernel and quantization coverage

This table tracks where inference primitives live after the backend reorganization.

Legend: implemented means the package owns the runtime or reference implementation; PTX means raw NVIDIA source constants only.

| Primitive / format | NVIDIA runtime | NVIDIA PTX | SIMD runtime | SIMD kernels | Vulkan | MLX backend |
|---|---|---|---|---|---|---|
| Device buffers / launch | `backends/nvidia/runtime` | — | — | — | `backends/vulkan` | — |
| SGEMM / F32 GEMV | `backends/nvidia/runtime` | `backends/nvidia/ptx/sgemm.go` | `backends/simd/runtime` | — | partial scaffold | — |
| Vector add/mul/scale | `backends/nvidia/runtime` | `backends/nvidia/ptx/vector.go` | `backends/simd/runtime` | — | vec-add only | — |
| Activations (SiLU/GELU fused) | `backends/nvidia/runtime`; standalone GELU N/A for current model hot paths | `backends/nvidia/ptx/activation.go` | `backends/simd/runtime` wrappers including scalar GELU compatibility | `backends/simd/kernels/activation.go` and `gelu.go` scalar kernels | partial scaffold | — |
| Norms | `backends/nvidia/runtime`; standalone LayerNorm N/A for current RMSNorm-based decoder paths | `backends/nvidia/ptx/norm.go` | `backends/simd/runtime` | `backends/simd/kernels/layernorm.go` | partial scaffold | — |
| Softmax | `backends/nvidia/runtime` row softmax; generic last-axis wrapper N/A until a non-attention model path needs it | `backends/nvidia/ptx/attention.go` | wrapper via kernels | `backends/simd/kernels/softmax.go` | — | — |
| RoPE | `backends/nvidia/runtime` | `backends/nvidia/ptx/attention.go` | `backends/simd/runtime` wrappers | `backends/simd/kernels/rope.go` scalar | partial scaffold | — |
| GQA attention | `backends/nvidia/runtime` | `backends/nvidia/ptx/attention.go` | wrapper via kernels | `backends/simd/kernels/attention.go` | — | — |
| BF16 | `backends/nvidia/runtime` | `backends/nvidia/ptx/bf16/` | `backends/simd/runtime/bf16` plus runtime facade | — | BF16 vec-add | — |
| Q4/GPTQ | `backends/nvidia/runtime` | `backends/nvidia/ptx/q4/` | `backends/simd/runtime/q4` | — | — | — |
| MLX affine | NVIDIA execution in `backends/nvidia/runtime` | `backends/nvidia/ptx/mlx/` | — | — | — | format/load/dequant/GEMV and scalar batched GEMV in `backends/mlx`; Switch-MoE expert loader validates U32/I32 packed weights plus BF16/F16/F32 scales/biases |
| NVFP4 | `backends/nvidia/runtime` | `backends/nvidia/ptx/nvfp4/` | `backends/simd/runtime/nvfp4` with scalar decode/dequant/GEMV, caller-owned dequant helper, and SIMD capability gates; Qwen loader rejects malformed packed shapes and non-16-aligned input dimensions | — | — | — |

See [bf16-parity.md](bf16-parity.md) for BF16 no-scale RMSNorm, LM-head, and NVIDIA-vs-CPU parity expectations. See [nvidia-quant-boundaries.md](nvidia-quant-boundaries.md) for NVIDIA Q4 asymmetric and NVFP4 packed/native support boundaries.

`runtime/quant` is a legacy compatibility package that delegates to backend-owned quantization packages. Repository model/backend code imports owning backends directly; new code should do the same unless deliberately maintaining external compatibility.
