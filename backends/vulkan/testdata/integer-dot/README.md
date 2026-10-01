# Explicit integer-dot diagnostic fixtures

These four shaders are test fixtures, not embedded production operators. Default shader admission refuses them. `VulkanInitIntegerDot` and `VkKernelCreateIntegerDot` explicitly enable the narrow packed mixed-signedness path; no Int8/storage8 feature is enabled. Physical devices require Vulkan 1.3, `shaderIntegerDotProduct` and accelerated packed mixed-signedness dot.

- `dot`: packed unsigned Q5-range bytes × signed bytes, exact int32 sum.
- `q8`: one invocation per block quantises 32 finite F32 values; F32 scale and inverse, GLSL `Round`, signed-byte packing, F16 scale and scaled integer sum.
- `q5q8`: independent-block Q5_0 × Q8_1 correction in F32.
- `linear`: tiled K32 Q5_0 × Q8_1 matrix multiply; F32 block correction, explicit FMA across blocks, F32 bias. It is a separately named approximate arithmetic diagnostic, not F32 parity.

Compile with shaderc v2026.1/glslang `301b4ede53d59b68bf55f95bb26412d9233c8187` (`glslc --target-env=vulkan1.1 -fshader-stage=compute`), strip debug with `spirv-opt --strip-debug`, then validate with `spirv-val --target-env vulkan1.3`. Offline glslang 12.0.0 lacks the integer-dot extension; it must not be used to regenerate these assets. Source and stripped binaries are hashed in `SHA256SUMS`.

SPIR-V grammar is pinned to SPIRV-Headers `vulkan-sdk-1.3.296.0`, SHA256 `83ee1c44ae5d87d54bb118be8ff041a9349661dd6c5ce0e02e1e4b539a6996bc`; Vulkan core header SHA256 `50af5a157c8aab7d90dcd929a05758b4dc3e78a619f46552bfd5cde67b2d46c1`. Optional envelope admits only packed capabilities 6018/6019, the exact `SPV_KHR_integer_dot_product` extension, typed six-word `OpSUDot`, and typed Round/FMax/half pack/unpack/conversions needed for quantisation. Debug metadata is removed instead of widening baseline opcode admission.

The scalar oracle is independent of GPU packing: it reads original 22-byte Q5 blocks and computes unsigned integer sum, then `dW * (sum * dQ - 16 * sQ)`. Native checks establish nearest-even ties on Sigma; GLSL `Round` leaves exact tie direction implementation-defined, so matching original-engine quantised bytes is still unproven. Original quantiser uses eight lanes per block and a different F32 maximum reduction/division arrangement. Finite synthetic blocks and F16 underflow/scaled sums are covered; nonfinite/overflow rejection and trained-model admission are open.

Native tests require explicit environment paths, physical-device match and timeout ≤120s. Operator diagnostics do not measure trained-model or whole-request speed. See [qualification report](../../../../docs/validation/vulkan-integer-dot-groundwork-20261001.md).
