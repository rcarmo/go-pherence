# Vulkan erf-form F32 GELU

`VkGELUErfF32` adds arena-backed GELU with exact in-place or disjoint output and ordered-plan composition. It uses an explicit erfc approximation of the erf form. The existing tanh GELU and CPU/model defaults are unchanged. No Vulkan device or trained model ran.

## Formula and arithmetic

For finite input `v`, the shader uses Abramowitz and Stegun, formula 7.1.26, with `a = abs(v)/sqrt(2)`, `t = 1/(1+0.3275911*a)` and:

```text
p = ((((1.061405429*t - 1.453152027)*t + 1.421413741)*t
       - 0.284496736)*t + 0.254829592)*t
q = p * exp(-a*a)
v < 0: 0.5*v*q
v >= 0: v*(1-0.5*q)
```

The negative branch avoids subtraction of nearly equal values. For `v <= -10`, return zero; for `v >= 10`, return `v`. This tail cutoff also bypasses overflow-prone arithmetic for extreme finite F32 inputs. It is not the overflow threshold. NaN/infinity inputs are outside the public contract; there is no content scan.

Source: [A&S page 299, formula 7.1.26](https://personal.math.ubc.ca/~cbm/aands/page_299.htm), [page scan](https://personal.math.ubc.ca/~cbm/aands/page_299.jpg). The consulted scan SHA256 is `ec184caf269b7905c640c2bba439b0f93ceeabd46cdbfead478a80efe78a90b5`. Mathematical coefficients were transcribed; no foreign runtime or implementation was copied.

All 20 floating add/subtract/multiply/divide instructions in the final embedded shader carry `NoContraction`; the offline contract test enforces this. Explicit F32 casts constrain the Go model's rounding boundaries. The model rounds Go's float64 `math.Exp` to F32. Vulkan `Exp`, division accuracy, subnormal handling and driver execution still need device qualification.

## Tensor and execution contract

- Two storage descriptors, four push bytes (`count`), 256×1×1 local size and no shared storage.
- Matching contiguous F32 shapes, ranks 1–8, exact byte extents, positive dimensions and checked shape products.
- Checked uint32 element count, queried workgroup count and storage-range limits. Group rounding uses uint64 before conversion.
- Exact input/output alias is permitted. Partial overlap is rejected. Each invocation reads and writes only its own element; padded tail invocations do not access storage. No cross-invocation dependency requires a shader barrier.
- `Stage` owns its tensor/push slices; generic stages are mutable and must not be altered to bypass operator checks.
- `Forward` and plans reuse the checked serialised lifetime lane. Closing an in-flight kernel or arena fails until confirmed drain.

## Verification

| Check | Result |
|---|---|
| Full offline selection | 108 top-level tests, 415 passing test/subtest events, no failures/skips |
| Shuffled repeats | 41 GELU/Linear/LayerNorm/plan/shader tests ×30; 4950 passing events, counted separately |
| Approximation fixtures | 139,939 finite inputs: uniform `[-10,10]`, seeded random F32 bit patterns, extrema, signed zero, subnormals and adjacent cutoff values |
| Stable float64 erfc oracle | Maximum absolute error `4.061483118711351e-7`, at input `3.1091666` |
| Current CPU `GELUExactScalar` | Maximum absolute difference `4.76837158203125e-7` |
| Source-model acceptance | `abs(error) <= 2e-6 + 2e-6*abs(reference)` against both references |
| Relative error diagnostic | Maximum `0.0010293551253219488` where `abs(reference)>1e-5`; small negative tails amplify relative error |
| Static validation | 14 embedded + 14 rebuilt modules pass `spirv-val --target-env vulkan1.3` |
| Rebuild comparison | All 14 narrowly normalised matches; GELU/Linear/LayerNorm/RoPE are byte-identical |
| Checker tests | Six Bun tests, 51 assertions |

Seven new top-level tests cover arithmetic, in-place/tail schedules, actual wrapper descriptor/push/grid capture, independent stage slices, rank/shape/range/alias admission, constructor/close and linear→in-place GELU plan retention after cancellation. Metadata-only tests cover ranks 1–8, uint32 count overflow and the maximum descriptor range without allocating large buffers. Admission tests reject an aligned partial overlap, malformed comparison arities, unexpected unary opcodes and an operand on `NoContraction`.

The GLSL.std.450 comment previously mislabelled opcode 27 as Tanh. It is `Exp`; Tanh is 21 and remains unadmitted. The inspector now admits `FAbs=4`, three ordered float comparisons and `NoContraction=42`. Grammar provenance and licence pointers are in [NOTICE-SPIRV-Headers](../../../backends/vulkan/NOTICE-SPIRV-Headers). This remains bounded metadata admission, not full semantic validation.

Focused make targets, speech/affine/media regressions, affected vet/builds and arm64 test cross-build pass. Full-tree and backend compile-only errors match the prior checkpoint. The race build cannot run because `gcc` is absent. The arm64 executable was not run.

The first delegated review timed out without findings. A smaller shader/wrapper review found no functional issue and requested a tail-cutoff comment clarification, which was applied. The parent then extended `precise` to the final return arithmetic and audited the resulting decorations; final tests and static validation passed. The smaller review did not inspect generated SPIR-V or surrounding barriers and does not cover device execution.

## Evidence and reproduction

- [evidence.json](evidence.json): counts, scope and source/evidence hashes.
- [Final static verification](static-final/verification.json) and [shader disassembly](gelu.spvasm).
- [Initial static verification](static/verification.json): superseded shader before extending final-output `NoContraction`; retained as historical evidence.
- [Offline test events](tests.jsonl), compressed repeat events and focused command logs in this directory.

```sh
make speech-vulkan-static-check VULKAN_SHADER_REPORT=/tmp/new-gelu-static
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-offline-check speech-affine-check speech-foundations-check speech-media-integration
```

Set `SPIRV_VAL` and `GLSLANG_VALIDATOR` if required. Use a new static report directory.

No GPU numerical fidelity, trained-model quality, speedup, quantised execution, resident encoder or device recovery is qualified. The strict SincNet four-failure hold is unchanged. No services, private audio, model weights, deployment, restart or push were used.
