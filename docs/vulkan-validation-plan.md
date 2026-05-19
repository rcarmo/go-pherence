# Vulkan validation plan

The Vulkan backend currently has loader/buffer scaffolding, embedded shader assets, generic compute dispatch machinery, and validating wrapper stubs for model-relevant primitives. Pipeline cache wiring and CPU-vs-Vulkan numeric parity remain the Phase 7 gap.

## Validation gates

1. **Device selection**
   - Default: reject CPU/software Vulkan devices such as llvmpipe.
   - Debug override: `GO_PHERENCE_VULKAN_ALLOW_CPU=1`.
   - Diagnostics: `GO_PHERENCE_VULKAN_DEBUG=1`.

2. **SPIR-V pipeline cache wiring**
   - Populate `initVkKernels` from embedded SPIR-V only after confirming shader modules create successfully on available drivers.
   - Keep every pipeline optional: a failed shader/pipeline should disable that operation without disabling the entire backend.
   - Preserve the existing validating wrappers, which already reject malformed dimensions/buffers before dispatch.

3. **Wrapper parity tests**
   - Use availability-gated tests so normal CPU-only CI skips cleanly when Vulkan is unavailable.
   - For each wrapper, compare against existing CPU scalar/SIMD owners:
     - RMSNorm/RMSNormNoScale → `backends/simd/runtime`
     - GEMV → dense CPU reference
     - SiLU/GELU fused activations → `backends/simd/kernels`
     - RoPEPartial → `backends/simd/kernels.ApplyRoPEPartial`
     - attention scores → CPU GQA score reference
   - Tolerances should initially be scalar-level for simple elementwise ops and relaxed only when hardware/driver behavior requires it.

## Current wrapper test coverage

`backends/vulkan/vulkan_wrapper_test.go` is availability-independent. It validates:

- malformed inputs return errors before dispatch,
- valid fake buffers return explicit pending-pipeline / not-available errors while pipeline cache entries are nil.

These tests are not a replacement for CPU-vs-Vulkan numeric parity; they only lock in API boundary behavior until real pipeline dispatch is enabled.

## Phase 7 remaining work

- Wire embedded SPIR-V into kernel cache.
- Add availability-gated CPU-vs-Vulkan numeric tests per wrapper.
- Keep Vulkan backend selection documentation current once dispatch is usable.
