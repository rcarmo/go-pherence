# Vulkan dispatch inventory

This inventory separates the legacy primitive wrapper surface from the resident F32 speech graphs. Availability of one primitive is not model readiness.

## Current package files

| Path | Purpose |
|---|---|
| `backends/vulkan/vulkan.go` | Loader/device initialization, queue/command-pool setup, CPU-device opt-in gate. |
| `backends/vulkan/vulkan_buf.go` | Vulkan buffer allocation/upload/download helpers. |
| `backends/vulkan/vulkan_dispatch.go` | Generic compute pipeline/descriptor/command dispatch scaffolding. |
| `backends/vulkan/vulkan_ops.go` | Legacy exported primitive wrappers and optional pipeline cache. |
| `backends/vulkan/vulkan_glsl.go` | GLSL source embedding/documentation helpers. |
| `backends/vulkan/vulkan_spirv.go` | Legacy hand-built vector-add module and pipeline-only loader; it exposes no placeholder model operation. |
| `backends/vulkan/vulkan_spirv_embedded.go` | Generated embedded SPIR-V blobs from `backends/vulkan/shaders/*.spv`; production operators consume these assets. |
| `backends/vulkan/debug.go` | Opt-in debug logging under `GO_PHERENCE_VULKAN_DEBUG`. |

## Shader assets present

| Shader | GLSL | SPIR-V | Planned wrapper |
|---|---:|---:|---|
| F32 vector add | yes | yes | `VkVecAddF32` wrapper with pipeline cache entry and parity test |
| BF16 vector add | yes | yes | `VkVecAddBF16` wrapper with pipeline cache entry; malformed-input coverage present |
| F32 RMSNorm | yes | yes | `VkRMSNormF32` wrapper with pipeline cache entry and parity test |
| BF16 RMSNorm | yes | yes | SPIR-V/pipeline cache entry present; exported wrapper pending until a CPU BF16 parity owner is needed |
| F32 RMSNormNoScale | yes | yes | `VkRMSNormNoScaleF32` wrapper with pipeline cache entry and parity test |
| F32 GEMV | yes | yes | `VkGemvF32` wrapper with pipeline cache entry and parity test |
| BF16 mixed GEMV | yes | yes | SPIR-V/pipeline cache entry present; exported wrapper pending until a CPU BF16 parity owner is needed |
| F32 SiLU×Mul | yes | yes | `VkSiLUMulF32` wrapper with pipeline cache entry and parity test |
| F32 GELU(tanh)×Mul | yes | yes | `VkGELUTanhMulF32` wrapper with pipeline cache entry and parity test |
| F32 RoPEPartial | yes | yes | `VkRoPEPartialF32` wrapper with pipeline cache entry and parity test |
| Attention scores | yes | yes | `VkAttentionScoresF32` wrapper with pipeline cache entry and parity test |

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
- `VkAttentionScoresF32(out, q, kCache *VkBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) error` — one workgroup per head×time pair; integral GQA head ratios only

The wrappers validate dimensions, buffer capacities, and product overflow before dispatch. `initVkKernels` now attempts to populate optional pipeline cache entries from embedded SPIR-V; a failed shader/pipeline disables only that operation and surfaces a not-available error. `backends/vulkan/vulkan_wrapper_test.go` covers invalid-input rejection and unavailable-pipeline errors without requiring a Vulkan device.

Availability-gated CPU-vs-Vulkan parity tests now cover `VkVecAddF32`, `VkRMSNormF32`, `VkRMSNormNoScaleF32`, `VkGemvF32`, `VkSiLUMulF32`, `VkGELUTanhMulF32`, `VkRoPEPartialF32`, and `VkAttentionScoresF32`.

## Resident speech graph status

The opt-in F32 Whisper graph owns fixed-shape Conv1D, add, LayerNorm, linear, erf-GELU and non-causal attention plans. The opt-in Community-1 graph owns fixed-shape LSTM sequence and ResNet Conv2D/channel-affine/add plans, with explicitly placed CPU SincNet/head/pooling/projection/postprocessing. Both use copied persistent weights and reject rather than silently fall back when their Vulkan contract fails.

Remaining work is model-specific: native/trained quality and placement qualification, quantised/F16 speech graphs, broader corpus evidence and process-level recovery after unrecoverable device state. `VulkanBF16Ready` only reports generic emulated-BF16 device admission; it does not claim that a BF16 speech graph exists.

## Next steps

1. Keep CPU/software Vulkan opt-in via `GO_PHERENCE_VULKAN_ALLOW_CPU`.
2. Add BF16 wrapper/model coverage only when an actual model path justifies it.
3. Keep backend selection and resident-graph documentation current with measured model evidence.

See [vulkan-validation-plan.md](vulkan-validation-plan.md) for the validation sequence and parity-test expectations.
