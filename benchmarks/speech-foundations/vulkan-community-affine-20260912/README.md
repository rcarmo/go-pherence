# Community-1 Vulkan channel affine and ReLU

`VkChannelAffineReLUF32` adds the first model-free Vulkan primitive selected for the Community-1 ResNet path. It applies a prepared per-channel affine transform to contiguous channel-major F32 tensors and optionally applies ReLU. CPU and model defaults are unchanged.

## Contract

The operation is:

```text
out[channel, spatial] = fma(input[channel, spatial], scale[channel], shift[channel])
if relu: out = max(out, +0)
```

A model owner prepares inference BatchNorm coefficients as `scale = weight / sqrt(runningVariance + epsilon)` and `shift = bias - runningMean*scale`. This generic Vulkan operator does not derive or own checkpoint policy. It supports rank-2–8 `[channels, spatial...]` tensors, 1–2048 channels, exact input/output alias or disjoint output, and shared read-only scale/shift storage. Partial input/output overlap and every coefficient/output overlap fail before recording.

The shader has four storage descriptors and 12 push bytes (`channels`, `spatial`, `relu`). One invocation owns one output element. The compiler emits one GLSL.std.450 `Fma`; ReLU uses ordered greater-than and assignment so `-0` becomes `+0`, matching Go `max(value,+0)` for finite inputs. The closed SPIR-V parser adds only ternary Fma and the emitted `OpINotEqual`/`OpFOrdGreaterThan` core comparison metadata. Inputs and prepared coefficients must be finite; there is no content scan or hidden fallback.

This slice does not implement WeSpeaker 2-D 3×3/1×1 convolution, upload a Community checkpoint, or add a model execution mode. Existing `VkAddF32` can supply residual addition after convolution and affine stages when the full graph is implemented.

## Offline verification

- Five focused operator tests pass. The schedule model covers 56,080 outputs across five channel/spatial geometries, affine-only and affine+ReLU, and four shuffled invocation orders. Its arithmetic oracle is `simd.FMA32Scalar`, including a known case that distinguishes native one-rounding F32 FMA from widened Go `float64` FMA.
- Tests cover exact in-place operation, disjoint output, copied stages, four descriptor ranges, push/group geometry, ranks 2–8, malformed shapes/extents, coefficient/output alias rejection, cancellation, grid bounds, constructor/close and plan retention through drain.
- Ten shuffled focused repetitions pass.
- The full mock-only Vulkan target passes.
- All 18 embedded and rebuilt shaders pass `spirv-val --target-env vulkan1.3`. The new stored, embedded and rebuilt SPIR-V bytes are identical (`53ec7f49bd5fb7344937f1c488a014ba2fdff9f577053b3866099deb440f116b`).
- The static checker passes six tests with 59 assertions.
- `go vet ./backends/vulkan` and the Linux/arm64 test cross-build pass.

A broad `go test ./backends/vulkan` invocation reached the existing availability-gated device parity suite and failed `TestVulkanAttentionScoresF32Parity` (`got 0 want 1.25`). That test is outside this operator and the mock-only gate passes. No further hardware parity was run.

The source-only external review timed out without findings or approval. Parent review checked shader indexing, integer products, aliasing, Fma/ReLU semantics, closed parser additions, stage ownership and pending-plan lifetime.

No Vulkan device, trained model, public/private audio, service, deployment, push or performance benchmark was used. Static inspection proves the shader contains the required FMA instruction but does not establish a device's floating-point controls or native output parity. The current CPU Community-1 BatchNorm uses widened Go `float64` FMA, so CPU/Vulkan bit identity is neither claimed nor expected at rare double-rounding boundaries. Community-1 Vulkan graph construction, 2-D convolution, model accuracy, device-loss recovery and whole-job placement remain open.
