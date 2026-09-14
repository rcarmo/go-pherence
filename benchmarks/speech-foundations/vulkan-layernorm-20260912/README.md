# Vulkan F32 row LayerNorm

`VkLayerNormF32` adds row-wise centred LayerNorm with learned scale and bias to the tensor arena and ordered F32 plan APIs. This checkpoint is static and mock-tested only; no Vulkan driver or shader was executed.

## Contract

- X/Out: identical contiguous `[rows,width]` F32 tensors, width 1–16384. Weight/Bias: `[width]` F32 tensors.
- One 256-invocation workgroup per row, 1024 bytes shared scratch, four storage descriptors and 12 push bytes (`rows`, `width`, `eps`).
- Two-pass centred population variance, with an explicit workgroup barrier after every lane loads the mean and before scratch reuse. This avoids the cancellation-prone `E[x²]-E[x]²` formula.
- Positive finite epsilon. Whisper uses `1e-5`; the wrapper permits other positive finite values.
- Exact X/Out range alias is permitted; partial overlap and output overlap with Weight/Bias are rejected. Read-only inputs may overlap.
- `Stage` returns copied binding/push arrays for `NewVkF32Plan`. `Forward` uses the same arena buffers without host intermediate transfers. Closure and timeout retention use the existing kernel/arena ownership rules.
- Input contents must be finite and have representable F32 reductions. The wrapper does not scan tensor contents or hide overflow behind a fallback. Numerical qualification for actual model activations is still required.

## Verification

- Seven new top-level tests; full offline suite: 94 top-level tests/399 passing events, zero failures/skips.
- Twenty-seven LayerNorm/plan/shader tests ×30 shuffled repeats: 4470 passing events, counted separately.
- Go schedule model: 11 widths (1 through 16384), three row patterns, four invocation permutations and exact-alias/separate-output comparison. Float64 centred-variance oracle comparison passes `2e-5 + 2e-5*abs(reference)` on these fixtures. Patterns include constant and low-variance rows.
- This schedule model does not execute SPIR-V or validate GPU rounding, denormals, extreme activations or all reductions.
- Mock calls cover successful construction/closure, four descriptor ranges, push ABI and row dispatch count, copied stages, rejected shapes/aliases/epsilon, two-stage plan execution and retained-owner cancellation.
- [Static check](static/verification.json): 12 embedded and 12 rebuilt modules pass `spirv-val --target-env vulkan1.3`. LayerNorm and repaired RoPE rebuild byte-identically; the other ten retain only the prior narrowly normalised differences. Checker tests: six tests, 47 assertions.
- Focused speech/affine/media regressions, Vulkan/board vet, affected builds and arm64 test cross-build pass. Full-tree/backend failures match the previous baseline. Race tests still cannot build because `gcc` is absent.

Narrow source review found no shader race, bounds, push or alias blocker. It raised a possible nil-kernel Close panic; the existing `VkComputeKernel.Close` accepts nil receivers, so that finding did not apply. A zero-value/nil operator regression test now covers the transitive contract. The new [disassembly](layernorm.spvasm) and [evidence manifest](evidence.json) are included.

## Reproduce

```sh
make speech-vulkan-static-check VULKAN_SHADER_REPORT=/tmp/new-layernorm-static
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-offline-check speech-affine-check speech-foundations-check speech-media-integration
```

Provide `SPIRV_VAL` and `GLSLANG_VALIDATOR` paths if the offline compiler tools are not on PATH. The report directory must be new.

No resident Whisper encoder, trained-model output, device numerical accuracy, performance gain, model-quality result or device recovery is qualified. Legacy model defaults are unchanged. SincNet's four strict failures remain. No GPU, trained checkpoint, private media, service, deployment, restart or push was used.
