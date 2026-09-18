# Vulkan tiled F32 linear projection

`VkLinearF32` implements arena-backed `X[M,K] * Weight[N,K]^T + Bias[N] -> Out[M,N]`, with direct `Forward` and ordered-plan `Stage` methods. No quantised format or trained encoder is wired here. No Vulkan device or shader execution was used for this checkpoint.

## Contract and implementation

- Four storage descriptors and 12 push bytes: `rows`, `inDim`, `outDim`.
- Workgroups are 16×16 invocations. Each workgroup produces a 16×16 output tile; one invocation owns each output element.
- Two flat shared float arrays contain the activation and weight tiles (2048 bytes total). Weight loads are contiguous along K; consumers index the weight tile by output column.
- All lanes reach the barriers. Invalid row/column/K-tail loads contribute zero; invalid output lanes do not store. A second barrier prevents the next tile from overwriting data still being read.
- M/N/K must each be 1–16384. Exact tensor ranks and byte extents, queried device limits and non-overlap of output with every input are checked before recording. Input/input alias is allowed. Bias is required; a zero bias tensor represents a no-bias projection.
- Caller tensor contents must have finite, representable F32 products/sums. There is no content scan or hidden fallback. Generic mutable stages and low-level custom dispatches remain the caller's responsibility.

## Verification

- Seven new top-level tests. Full offline selection: 101 top-level tests, 407 passing test/subtest events, zero failures/skips.
- Thirty-four Linear/LayerNorm/plan/shader tests ×30 shuffled repeats: 4710 passing events, counted separately.
- A Go tile-schedule model compares ten shapes ×four shuffled workgroup/lane schedules against serial F32 and independent float64 dot products. Odd M/N/K, tile boundaries and K up to 16384 are covered. Every output has one writer. Model outputs are bit-identical to the serial F32 schedule on these fixtures and within `2e-5 + 2e-5*abs(reference)` of the float64 oracle.
- An analytic 2×3 by 2×3 case checks weight orientation and bias. No SPIR-V was executed by the model; GPU contraction/rounding and performance are unqualified.
- Mock wrapper capture checks descriptor order/ranges, push words, two-dimensional group geometry and copied stages. Invalid ranks, extents, zero/oversized axes, aliasing and device-grid limits reject before native mutation.
- A linear→LayerNorm ordered plan verifies cancellation retains both operators and both backing arenas until confirmed drain. Successful constructor/close and metadata resource admission are tested.
- [Static verification](static/verification.json): 13 embedded and 13 rebuilt modules pass `spirv-val --target-env vulkan1.3`. The new linear shader, LayerNorm and repaired RoPE rebuild byte-identically. Other modules retain the prior narrowly normalised differences. Checker: six Bun tests, 49 assertions.
- Focused speech/affine/media regressions, Vulkan/board vet, affected builds and arm64 test cross-build pass. Full-tree/backend failures match the previous baseline. The race build still lacks `gcc`.

Narrow source review found no scoped indexing, barrier, overflow, push or alias bug. It assumed the embedded binary matched the shader; the static rebuild check verifies that byte identity. An initial admission test lacked a mocked push entry point; this test setup was corrected before the final passing runs.

[evidence.json](evidence.json) records source/evidence hashes and verification scope. The [shader disassembly](linear.spvasm) is included. Reproduce with:

```sh
make speech-vulkan-static-check VULKAN_SHADER_REPORT=/tmp/new-linear-static
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-offline-check speech-affine-check speech-foundations-check speech-media-integration
```

Set `SPIRV_VAL` and `GLSLANG_VALIDATOR` if the offline tools are not on PATH. Static report directories must be new.

No GPU numerical fidelity, trained-model quality, speedup, quantised GEMM, resident Whisper graph or device recovery is qualified. SincNet's strict four-failure hold is unchanged. No GPU, trained checkpoint, private media, service, deployment, restart or push was used.
