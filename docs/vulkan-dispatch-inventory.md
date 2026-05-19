# Vulkan dispatch inventory

Phase 7 tracks what the Vulkan backend already owns versus what still needs safe runtime wrappers.

## Current package files

| Path | Purpose |
|---|---|
| `backends/vulkan/vulkan.go` | Loader/device initialization, queue/command-pool setup, CPU-device opt-in gate. |
| `backends/vulkan/vulkan_buf.go` | Vulkan buffer allocation/upload/download helpers. |
| `backends/vulkan/vulkan_dispatch.go` | Generic compute pipeline/descriptor/command dispatch scaffolding. |
| `backends/vulkan/vulkan_ops.go` | Initial exported operation wrappers; currently vec-add focused with comments for additional kernels. |
| `backends/vulkan/vulkan_glsl.go` | GLSL source embedding/documentation helpers. |
| `backends/vulkan/vulkan_spirv.go` | Hand-built/placeholder SPIR-V helpers. |
| `backends/vulkan/vulkan_spirv_embedded.go` | Generated embedded SPIR-V blobs from `backends/vulkan/shaders/*.spv`. |
| `backends/vulkan/debug.go` | Opt-in debug logging under `GO_PHERENCE_VULKAN_DEBUG`. |

## Shader assets present

| Shader | GLSL | SPIR-V | Planned wrapper |
|---|---:|---:|---|
| F32 vector add | yes | yes | existing `VkVecAddF32` wrapper, but kernel init still marked pending validation |
| BF16 vector add | yes | yes | existing `VkVecAddBF16` wrapper, but kernel init still marked pending validation |
| F32 RMSNorm | yes | yes | pending `VkRMSNormF32` wrapper |
| BF16 RMSNorm | yes | yes | pending `VkRMSNormBF16` wrapper |
| F32 RMSNormNoScale | yes | yes | pending `VkRMSNormNoScaleF32` wrapper |
| F32 GEMV | yes | yes | pending `VkGemvF32` wrapper |
| BF16 mixed GEMV | yes | yes | pending `VkGemvBF16Mixed` wrapper |
| F32 SiLU×Mul | yes | yes | pending `VkSiLUMulF32` wrapper |
| F32 GELU(tanh)×Mul | yes | yes | pending `VkGELUTanhMulF32` wrapper |
| F32 RoPEPartial | yes | yes | pending `VkRoPEPartialF32` wrapper |
| Attention scores | yes | yes | pending `VkAttentionScoresF32` wrapper |

## Exported wrapper status

Public wrapper functions now exist for:

- `VkVecAddF32(dst, a, b *VkBuf, n int) error`
- `VkVecAddBF16(dst, a, b *VkBuf, n int) error`
- `VkRMSNormF32(x, w *VkBuf, n int, eps float32) error`
- `VkRMSNormNoScaleF32(x *VkBuf, n int, eps float32) error`
- `VkGemvF32(out, x, w *VkBuf, inDim, outDim int) error`
- `VkSiLUMulF32(dst, gate, up *VkBuf, n int) error`
- `VkGELUTanhMulF32(gate, up *VkBuf, n int) error`
- `VkRoPEPartialF32(x, freqs *VkBuf, pos, nHeads, headDim, rotHalf int) error`
- `VkAttentionScoresF32(out, q, kCache *VkBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) error`

The newly added wrappers validate dimensions, buffer capacities, and product overflow before dispatch, then return a clear "pipeline wiring pending" unsupported error while kernel cache entries remain unpopulated. `backends/vulkan/vulkan_wrapper_test.go` covers invalid-input rejection and pending-pipeline errors without requiring a Vulkan device. `VkVecAddF32`/`VkVecAddBF16` predate this validation pass and still depend on kernel cache entries that are not populated because `initVkKernels` intentionally logs SPIR-V validation as pending instead of constructing pipelines.

## Next Phase 7 steps

1. Populate kernel cache from embedded SPIR-V only after validating the generated binaries on available drivers.
2. Add safe wrapper functions one primitive at a time with dimension/product checks before dispatch.
3. Keep CPU/software Vulkan opt-in via `GO_PHERENCE_VULKAN_ALLOW_CPU`.
4. Add CPU-vs-Vulkan tests behind availability/opt-in gates for each wrapper.
