# Community-1 resident Vulkan BasicBlock owner

`community1.NewVulkanBasicBlock` composes the checked Vulkan CHW convolution, channel-affine/ReLU and residual-add primitives into one fixed-shape WeSpeaker ResNet BasicBlock owner. This is an explicit model API; no existing CPU mode, model path, service profile or default selects it.

## Ownership and graph

Construction first validates the complete CPU-owned `WeSpeakerBasicBlock` and input geometry without a Vulkan call. It prepares and copies BatchNorm scale/shift vectors with the existing Community-1 inference formula, adds immutable identity/zero coefficients for the final ReLU, snapshots tensor shapes and computes an aligned arena extent. Only then does it query the initialized device and create operators, one arena, uploaded tensors and one private plan.

Identity blocks use six stages:

```text
conv3 → affine+ReLU → conv3 → affine → residual add(input) → ReLU
```

Projection blocks use eight stages by inserting `conv1(stride) → affine` for the shortcut before the residual add. Two output-shaped scratch tensors alternate safely; the final output is always the second. Every source parameter is copied before native allocation, and no caller slice is retained.

Calls are serialized. `Forward` validates finite exact-length CHW input, uploads once, executes the resident plan once and downloads once. `Close` first prevents new calls, closes plan/arena/operators in reverse ownership order and retains failed resources for retry. It never hides an in-flight submission, drains the device, restarts a service or falls back to CPU.

## Model-free verification

- All pinned synthetic block fixtures produce valid layouts, including identity and projected shortcuts.
- Tests verify 6/8-stage topology, every read-after-definition, exact CHW and weight shapes, prepared BatchNorm coefficients, immutable ReLU coefficients, alignment accounting and source-copy independence.
- Invalid context/source/shape, nonfinite parameters, negative variance, arithmetic overflow and invalid plan constructor fail before device access.
- Shared-copy lifecycle tests verify permanent stop after partial close, reverse retry ownership, idempotence and context-bounded serialization.
- The focused Community Vulkan gate also passes convolution/affine ABI and the 19-shader inventory. Separate final gates run the complete mock-only Vulkan suite, ten shuffled convolution repetitions, vet, arm64 cross-build and static validation.

No Vulkan device, trained checkpoint, audio, full ResNet34 graph, pooling/projection, service, deployment, push or performance run was used. Native block parity, constructor rollback against real Vulkan resources, device loss, complete trunk residency and CPU/full-Vulkan/hybrid whole-job comparisons remain open.
