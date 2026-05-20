# Backend selection

Backend selection is intentionally conservative: model execution should prefer the fastest validated backend, but every backend boundary must fail closed to a checked CPU/SIMD or scalar path rather than performing unsafe/device work with malformed inputs.

## Selection order

1. **NVIDIA PTX** — selected for production GPU execution when NVIDIA is enabled, initialized, and the requested primitive has a validating runtime wrapper.
2. **Vulkan SPIR-V** — portable GPU dispatch for explicit Vulkan wrapper calls and availability-gated tests. Software Vulkan devices are rejected by default and are only allowed with `GO_PHERENCE_VULKAN_ALLOW_CPU=1`.
3. **CPU SIMD runtime** — AVX2/FMA on amd64 and NEON on arm64 where implemented, with runtime capability gates.
4. **Go scalar/reference kernels** — universal fallback and correctness oracle for parity tests.

The current model forward path still treats Vulkan as explicit/experimental rather than an automatic replacement for CPU fallback. Vulkan wrappers are usable for their covered primitives, but callers should keep CPU fallback logic until full model-level Vulkan placement is introduced.

## Backend gates

| Gate | Effect |
|---|---|
| `GO_PHERENCE_DISABLE_NVIDIA=1` | Forces CPU/non-NVIDIA behavior for tests and diagnostics. |
| `GO_PHERENCE_GPU_DEBUG=1` | Enables NVIDIA init/module/stream/native-BF16/direct-ioctl diagnostics. |
| `GO_PHERENCE_VULKAN_DEBUG=1` | Enables Vulkan discovery, device creation, shader, and wrapper diagnostics. |
| `GO_PHERENCE_VULKAN_ALLOW_CPU=1` | Allows software Vulkan devices such as llvmpipe for shader debugging. Disabled by default. |
| `GO_PHERENCE_LOAD_DEBUG=1` | Enables loader, quantization, placement, LM-head, expert-pool, and VRAM budget diagnostics. |
| `GO_PHERENCE_PREFILL_DEBUG=1` | Enables batched-prefill progress diagnostics. |

## Vulkan wrapper status

`backends/vulkan` now has validating dispatch wrappers and embedded SPIR-V pipeline-cache wiring for:

- `VkVecAddF32` / `VkVecAddBF16`
- `VkRMSNormF32` / `VkRMSNormBF16`
- `VkRMSNormNoScaleF32`
- `VkGemvF32` / `VkGemvBF16Mixed`
- `VkSiLUMulF32`
- `VkGELUTanhMulF32`
- `VkRoPEPartialF32`
- `VkAttentionScoresF32`

Each wrapper validates dimensions and buffer sizes before dispatch. Availability-gated CPU-vs-Vulkan tests cover the exported wrappers; the tests skip cleanly when Vulkan is unavailable and keep software devices opt-in.

## Fallback rules

- Backend wrappers must validate dimensions, byte-size products, nil buffers, and output capacity before unsafe or device calls.
- A missing optimized backend path is not an error when a documented scalar/SIMD fallback exists.
- Hardware-gated tests should skip when the backend is unavailable, but malformed-input tests should run without hardware wherever possible.
- Documentation tables in `kernel-coverage.md`, `gpu-options.md`, and `vulkan-dispatch-inventory.md` should be updated whenever wrapper coverage changes.
