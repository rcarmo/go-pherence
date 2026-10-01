# Explicit integer-dot fixtures and candidates

Default shader admission refuses these optional arithmetic modules. `VulkanInitIntegerDot` and `VkKernelCreateIntegerDot` explicitly enable packed mixed-signedness dot; no Int8/storage8 feature is enabled. Devices require Vulkan1.3, the integer-dot feature and accelerated packed mixed-signedness execution.

- `dot`: packed unsigned Q5-range bytes × signed bytes, exact int32 sum.
- `q8`: serial maximum over 32 finite F32 activations; compiled-original operation order `d=amax*(1/127)`, `inv=127/amax`, GLSL Round, F16 scale/scaled sum and signed-byte packing. Invalid blocks emit a NaN marker. Magnitude above1000, nonfinite values or nonzero maxima below1e-30 are rejected this way.
- `q5q8`: independent-block Q5_0×Q8_1 correction in F32.
- `linear`: K32 tiled Q5_0×Q8_1, F32 correction and explicit FMA across blocks, then F32 bias. This is not ordered F32 per-element arithmetic.
- `q8-coop`: the same Q8_1 words as `q8`, computed by eight invocations per block with coalesced loads; maximum, invalid flags and integer sums are order-independent.
- `linear-mmq`: the `linear` per-output arithmetic (integer block sums, precise `dot*ds.x-16*ds.y`, block-ordered FMA, bias last) on the original MMQ schedule: 128 invocations, 64×64 tile, four K blocks per shared-memory stage, register-resident Q5 unpack. Requires K%128==0.

The explicit experimental `VkLinearQ5IntegerDotSet` embeds `q8` and `linear` (or `q8-coop` and `linear-mmq` through `NewVkLinearQ5IntegerDotMMQSetStream`) from `shaders/integer-dot/`; matching diagnostic copies remain here. The mode is not selected by defaults. Its trained Portuguese segment end regresses, so it has no overall timing/accuracy acceptance.

Compile using shaderc v2026.1/glslang `301b4ede53d59b68bf55f95bb26412d9233c8187` with `glslc --target-env=vulkan1.1 -fshader-stage=compute`; strip debug via `spirv-opt --strip-debug`; validate with `spirv-val --target-env vulkan1.3`. Offline glslang12 does not support the integer-dot extension. Source/binaries are hashed in `SHA256SUMS`; `scripts/check-vulkan-integer-dot.ts` independently checks regeneration and fixture equality. Existing28 production shaders retain their separate gate.

SPIR-V grammar pin: SPIRV-Headers `vulkan-sdk-1.3.296.0`, SHA256 `83ee1c44ae5d87d54bb118be8ff041a9349661dd6c5ce0e02e1e4b539a6996bc`; Vulkan core header SHA256 `50af5a157c8aab7d90dcd929a05758b4dc3e78a619f46552bfd5cde67b2d46c1`. Optional admission requires packed capabilities6018/6019, the exact integer-dot extension, typed six-word `OpSUDot` and narrowly typed quantisation instructions. It is metadata validation, not semantic/safety proof.

The scalar oracle reads original22-byte Q5 blocks independently. Original compiled ×4 subgroup/non-subgroup quantisers match all bytes/scales/sums for300,000 actual layer0 blocks after the inverse-scale correction. GLSL Round tie behaviour is implementation-defined; Sigma's nearest-even diagnostic passes. Other layers/devices and complete original MMQ accumulation remain unqualified.

Native tests require explicit paths, pinned input/model hashes, physical-device match and timeout≤120s. See [groundwork](../../../../docs/validation/vulkan-integer-dot-groundwork-20261001.md) and [trained candidate report](../../../../docs/validation/whisper-integer-dot-ffn-20261001.md).
