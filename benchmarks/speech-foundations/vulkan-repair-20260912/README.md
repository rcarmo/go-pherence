# No-scale RMSNorm and RoPE repair

The two source/interface holds found by the earlier [static audit](../vulkan-static-20260912/README.md) are repaired. Device numerical validation is still required; no Vulkan driver or shader was executed.

## Changes

- `VkRMSNormNoScaleF32`: two descriptors bind the exact same buffer/range, one workgroup. The reduction shader and its binary are unchanged. Added explicit uint32 admission for `n`.
- `rope_partial_f32.glsl`: one invocation reads and writes both components of a disjoint pair. No other invocation accesses that pair; untouched tail dimensions remain unchanged. Q and frequency buffers must not alias.
- RoPE frequencies use the existing interleaved `[position, pair, cos/sin]` API. Push order is `Pos, Heads, HeadDim, RotHalf`. Host dispatch uses `ceil(heads*rotHalf/256)` workgroups, with checked uint32 extents for shader indexing.
- Regenerated only the RoPE stored `.spv` and Go embedded byte array. Existing layout checks now admit all 11 wrapper requests; they still reject genuinely undersized descriptor/push layouts.

## Verification

- Six new top-level mock/source-model tests. Full selection: 87 top-level tests, 391 passing test/subtest events, zero failures/skips.
- Seventeen relevant tests × 30 shuffled repeats: 3330 passing events, counted separately.
- Three-second host geometry fuzz: 118220 inputs, pass.
- Five RoPE geometry cases × eight invocation-order permutations match the pre-existing CPU reference bit-for-bit; every rotated component has one writer, tails have none. Includes odd dimensions and workgroup-tail/large rotation cases.
- Seven RMS lengths × four output-invocation permutations match separate-output computation using the same modelled reduction. This is a Go schedule model, not SPIR-V execution or a hardware tolerance result.
- Captured actual wrapper calls confirm exact descriptors, push words and group dimensions. Reject tests cover aliasing, short tensors, overflowing positions and too-wide RMS `n`.
- [Static verification](static/verification.json): 11 embedded modules and 11 rebuilt modules pass `spirv-val --target-env vulkan1.3`. All stored binaries match embedded bytes; the new RoPE rebuild is byte-identical. The other ten retain generator/decoration-order-only differences.
- Static checker: six Bun tests, 45 assertions. Focused speech regressions, Vulkan/board vet, affected builds and arm64 test cross-build pass. Full-tree/backend failures match the previous baseline; the race build still lacks `gcc`.

Narrow source review found no scoped race/ABI/bounds blocker. It identified the older RMS push narrowing issue; the explicit bound and regression test were added before final verification. The original audit, including the old RoPE disassembly, remains unchanged.

## Reproduce

With local offline compiler tools on PATH (or `SPIRV_VAL`/`GLSLANG_VALIDATOR` set):

```sh
make speech-vulkan-static-check VULKAN_SHADER_REPORT=/tmp/new-repair-check
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-offline-check speech-affine-check speech-foundations-check speech-media-integration
```

[evidence.json](evidence.json) records source/evidence hashes and exact scope. The new [RoPE disassembly](rope-repaired.spvasm) is included for inspection. The package exposes low-level custom-kernel APIs; their callers must independently obey numeric types, shapes, alias restrictions and shader dispatch geometry.

No GPU numerical fidelity, trained-model quality, performance improvement or end-to-end speech result is established here. No native loader, trained checkpoint, private audio, service, deployment, restart or push was used. SincNet's four strict failures remain unchanged.
