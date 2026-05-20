# Vulkan validation plan

The Vulkan backend currently has loader/buffer scaffolding, embedded shader assets, generic compute dispatch machinery, validating wrappers for model-relevant F32 primitives, optional embedded-SPIR-V pipeline cache entries, and availability-gated CPU-vs-Vulkan numeric parity tests.

## Validation gates

1. **Device selection**
   - Default: reject CPU/software Vulkan devices such as llvmpipe.
   - Debug override: `GO_PHERENCE_VULKAN_ALLOW_CPU=1`.
   - Diagnostics: `GO_PHERENCE_VULKAN_DEBUG=1`.

2. **SPIR-V pipeline cache wiring**
   - `initVkKernels` populates optional kernels from embedded SPIR-V.
   - Keep every pipeline optional: a failed shader/pipeline disables that operation without disabling the entire backend.
   - Preserve validating wrappers, which reject malformed dimensions/buffers before dispatch.

3. **Wrapper parity tests**
   - Use availability-gated tests so normal CPU-only CI skips cleanly when Vulkan is unavailable.
   - Covered wrappers compare against existing CPU scalar/SIMD owners:
     - RMSNorm/RMSNormNoScale → `backends/simd/runtime`
     - GEMV → dense CPU reference
     - SiLU/GELU fused activations → `backends/simd/kernels`
     - RoPEPartial → `backends/simd/kernels.ApplyRoPEPartial`
     - attention scores → CPU GQA score reference
   - Tolerances should stay scalar-level for simple elementwise ops and relax only when hardware/driver behavior requires it.

## Current wrapper test coverage

`backends/vulkan/vulkan_wrapper_test.go` is availability-independent. It validates:

- malformed inputs return errors before dispatch,
- valid fake buffers return explicit pending-pipeline / not-available errors while pipeline cache entries are nil.

These tests complement CPU-vs-Vulkan numeric parity tests in `vulkan_parity_test.go` and `vulkan_more_parity_test.go`.

## Phase 7 remaining work

- Keep BF16 Vulkan wrapper/export parity scoped to a future CPU BF16 owner or model-level need.
- Keep Vulkan backend selection documentation current as dispatch and placement evolve.
