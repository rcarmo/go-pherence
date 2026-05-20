# Malformed-input coverage tracker

This tracker records exported backend/model wrapper malformed-input coverage added during the backend coverage work. It is not a replacement for package-local tests; it is an index for Phase 10 acceptance.

## Covered or staged for validation

| Area | Package/file | Coverage |
|---|---|---|
| SIMD RoPE | `backends/simd/kernels/rope_test.go`, `backends/simd/runtime/rope_test.go`, `rope_freqs_test.go` | Negative positions, empty/short freqs, zero heads/dims, odd head dims, head/tail preservation, frequency-table overflow/bad-theta guards, runtime wrapper smoke. |
| SIMD dense/softmax/norm/attention | `backends/simd/runtime/gemv_test.go`, `sgemm_checked_test.go`, `softmax_test.go`, `layernorm_test.go`, `attention_test.go`, `import_boundary_test.go`; `tensor/tensor_test.go`; `models/bert/bert_test.go` | Dense GEMV/GEMM, SGEMM, and row-bias zero-dimension, short-buffer, leading-dimension, tail-preservation, product-overflow, stride-overflow, and tensor Linear checked-path tests; softmax and LayerNorm nil/non-finite/row-shape/last-axis/affine/overflow/aliasing guards plus tensor checked-path, unsafe-slice, and SIMD import-boundary validation; checked GQA attention caller-owned/custom-scale/default-scale short-buffer/head-group/overflow guards; BERT checked linear and MHA helper no-op behavior plus fast/regular MHA softmax parity and BLAS import-boundary coverage; non-SIMD packages are guarded against direct kernel imports and unsafe SGEMM calls; model-family/tensor code is guarded against `runtime/quant` compatibility imports. |
| SIMD activations | `backends/simd/kernels/activation_test.go`, `backends/simd/runtime/activation_checked_test.go`; `tensor/tensor_test.go` | Short inputs preserve destination tails; checked SiLU/GELU entrypoints reject malformed inputs; tensor GELU zero-sized path covered; scalar golden tolerance names are explicit. |
| Q4/GPTQ | `backends/simd/quant/q4/*_test.go` | ValidateGemvSym malformed inputs, group-index/scale shape errors, scalar GEMV vs dequant reference, caller-owned dequant tail preservation, capability gates disabled until SIMD kernels land. |
| MLX | `backends/mlx/*_test.go` | Loader dtype/shape checks, in-memory quant validation, caller-owned dequant tail preservation, scalar batched GEMV malformed input rejection, capability gates disabled until SIMD kernels land. |
| NVFP4 CPU | `backends/simd/quant/nvfp4/nvfp4_test.go`, `capabilities_test.go` | FP4/F8 decode, malformed weight validation, caller-owned dequant tail preservation, explicit scalar GEMV reference, capability gates disabled until SIMD/native kernels land. |
| NVIDIA BF16/Q4/NVFP4/runtime buffers | `backends/nvidia/runtime/*_test.go` | BF16 LM-head byte-size overflow/short-buffer checks, Q4 upload and buffer sizing, NVFP4 buffer validation and sizing helpers, malformed DevBuf length sanitization, CUDA launch guards, and PTX loader empty-input rejection. |
| Qwen NVFP4/RoPE/native-MTP helpers | `model/qwen/*_test.go`, `loader/config/config_test.go` | NVFP4 packed-shape/group alignment validation; RoPE frequency allocation overflow and bad-theta fallback; bounded native-MTP required tensor enumeration. |
| Shared model RoPE | `model/rope_test.go` | RoPE frequency allocation overflow/bad dims/bad theta fallback. |
| Shared model/BERT/tensor/loader/KV/ioctl helpers | `model/checked_test.go`, `models/bert/bert_test.go`, `tensor/tensor_test.go`, `loader/safetensors/safetensors_test.go`, `runtime/kv/cache_test.go`, `backends/nvidia/ioctl/helpers_test.go` | Checked product/multiplication/addition/shape helpers reject negative and overflowing dimensions/offsets before allocation/slicing/file-range guards; KV compressed-entry and compressed bytes-per-head guards reject invalid dimensions. |
| MoE loaders/forward | `model/moe_test.go`, `model/moe_gpu_test.go` | Switch-MLX dtype handling, malformed MoE inputs, incomplete expert weight handling, GPU active-expert clamping, and expert-pool size overflow rejection before native uploads. |
| Vulkan wrappers | `backends/vulkan/vulkan_wrapper_test.go`, `vulkan_kernel_create_test.go`, `vulkan_buf_guard_test.go`, `vulkan_spirv_test.go` | Wrapper-level dimension/buffer rejection, kernel creation guards, buffer transfer guards, SPIR-V loader input validation, and pending-pipeline errors without requiring a Vulkan device. |

## Remaining Phase 10 gaps

- Add availability-gated NVIDIA parity tests where hardware is available.
- Extend availability-gated Vulkan CPU-vs-Vulkan numeric tests only when new wrappers land; F32 covered wrappers already have parity tests.
- Add AVX2/NEON parity tests when actual SIMD kernels land for RoPE, activations, Q4, MLX, and NVFP4.
- Keep new exported backend wrappers paired with package-local malformed-input tests.
