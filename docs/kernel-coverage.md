# Kernel and quantization coverage

This table tracks where inference primitives live after the backend reorganization.

Legend: implemented means the package owns the runtime or reference implementation; PTX means raw NVIDIA source constants only.

| Primitive / format | NVIDIA runtime | NVIDIA PTX | SIMD runtime | SIMD kernels | Vulkan | MLX backend |
|---|---|---|---|---|---|---|
| Device buffers / launch | `backends/nvidia/runtime` | — | — | — | `backends/vulkan`; see `docs/vulkan-dispatch-inventory.md` for shader/wrapper inventory | — |
| SGEMM / F32 GEMV | `backends/nvidia/runtime` | `backends/nvidia/ptx/sgemm.go` | `backends/simd/runtime` dense GEMV/GEMM references plus checked SGEMM slice APIs; model/tensor/BERT callers use checked entrypoints | — | validating F32 GEMV wrapper with embedded SPIR-V pipeline-cache wiring and availability-gated parity | — |
| Vector add/mul/scale/bias | `backends/nvidia/runtime` | `backends/nvidia/ptx/vector.go` | `backends/simd/runtime` including checked row-bias helper for dense outputs | — | vec-add wrapper present | — |
| Activations (SiLU/GELU fused) | `backends/nvidia/runtime`; standalone GELU N/A for current model hot paths | `backends/nvidia/ptx/activation.go` | `backends/simd/runtime` wrappers including checked scalar GELU compatibility; tensor and BERT GELU use runtime APIs | `backends/simd/kernels/activation.go` and `gelu.go` scalar kernels | validating SiLU×Mul and GELU×Mul wrappers with embedded SPIR-V pipeline-cache wiring and availability-gated parity | — |
| Norms | `backends/nvidia/runtime`; standalone LayerNorm N/A for current RMSNorm-based decoder paths | `backends/nvidia/ptx/norm.go` | `backends/simd/runtime` RMSNorm plus checked last-axis LayerNorm with aliasing coverage; tensor and BERT LayerNorm use runtime APIs | `backends/simd/kernels/layernorm.go` | validating RMSNorm/RMSNormNoScale wrappers with embedded SPIR-V pipeline-cache wiring and availability-gated parity | — |
| Softmax | `backends/nvidia/runtime` row softmax; generic last-axis wrapper N/A until a non-attention model path needs it | `backends/nvidia/ptx/attention.go` | `backends/simd/runtime` in-place, row-wise, and checked last-axis references; tensor/BERT regular and fast attention softmax use runtime APIs | `backends/simd/kernels/softmax.go` | — | — |
| RoPE | `backends/nvidia/runtime` | `backends/nvidia/ptx/attention.go` | `backends/simd/runtime` wrappers plus model/Qwen RoPE frequency table generation | `backends/simd/kernels/rope.go` scalar | validating RoPEPartial wrapper with embedded SPIR-V pipeline-cache wiring and availability-gated parity | — |
| GQA attention | `backends/nvidia/runtime` | `backends/nvidia/ptx/attention.go` | `backends/simd/runtime` wrappers including checked caller-owned, custom-scale allocated, and default-scale allocated APIs | `backends/simd/kernels/attention.go` | validating attention-score wrapper with embedded SPIR-V pipeline-cache wiring and availability-gated parity | — |
| BF16 | `backends/nvidia/runtime` | `backends/nvidia/ptx/bf16/` | `backends/simd/runtime/bf16` plus runtime facade | — | BF16 vec-add | — |
| Q4/GPTQ | `backends/nvidia/runtime` | `backends/nvidia/ptx/q4/` | `backends/simd/runtime/q4` | — | — | — |
| MLX affine | NVIDIA execution in `backends/nvidia/runtime` | `backends/nvidia/ptx/mlx/` | — | — | — | format/load/dequant/GEMV and scalar batched GEMV in `backends/mlx`; Switch-MoE expert loader validates U32/I32 packed weights plus BF16/F16/F32 scales/biases |
| NVFP4 | `backends/nvidia/runtime` | `backends/nvidia/ptx/nvfp4/` | `backends/simd/runtime/nvfp4` with scalar decode/dequant/GEMV, caller-owned dequant helper, and SIMD capability gates; Qwen loader rejects malformed packed shapes and non-16-aligned input dimensions | — | — | — |

See [backend-parity-matrix.md](backend-parity-matrix.md) for scalar/reference parity targets. See [bf16-parity.md](bf16-parity.md) for BF16 no-scale RMSNorm, LM-head, and NVIDIA-vs-CPU parity expectations. See [nvidia-quant-boundaries.md](nvidia-quant-boundaries.md) for NVIDIA Q4 asymmetric and NVFP4 packed/native support boundaries. See [final-coverage-acceptance.md](final-coverage-acceptance.md) for final acceptance criteria.

`runtime/quant` is a legacy compatibility package that delegates to backend-owned quantization packages. Repository model/backend code imports owning backends directly; new code should do the same unless deliberately maintaining external compatibility.
