# Vulkan reusable command-buffer reset — 12 September 2026

Baseline: `de133773a1dcceb61957a7a7a608676ec673cbb1` on `feat/speech-simd-vulkan`.

## Finding

The global command pool is created with `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`, but reusable single-kernel and `VkF32Plan` command buffers were passed directly to `vkBeginCommandBuffer` on each call. That relies on successful prior executable-state reuse and does not recover a command buffer left in an invalid recording state after `vkEndCommandBuffer` fails.

## Change

- Added `vkResetCommandBuffer` to the mandatory Vulkan symbol inventory.
- Kernel construction now requires reset support before native allocation.
- Single-dispatch and multi-stage plan execution call `vkResetCommandBuffer(command, 0)` before descriptor mutation and recording.
- Context cancellation is checked again after reset and before descriptor writes.
- Reset failures return the native result without descriptor updates, command recording, fence reset, submission or pending-resource publication.
- Existing fence reset remains immediately before queue submission; command reset and fence reset are distinct operations in tests and diagnostics.

No shader, dispatch geometry, memory barrier, tensor layout, model graph, default or numerical tolerance changed.

## Verification

All checks are mock-only; no Vulkan loader/device/GPU was opened.

- Full `backends/vulkan` suite passes.
- New single-dispatch and plan tests inject `vkEndCommandBuffer` failure, then verify a second call resets and records/submits successfully.
- Reset failure tests verify no descriptor or later native side effects.
- Initialization's every-missing-symbol test includes `vkResetCommandBuffer`.
- Command/barrier ordering tests include command reset before descriptor mutation.
- Linux/arm64 `backends/vulkan` test-binary cross-build passes.

Dependent Whisper/Community model-free tests, shuffled repetitions, vet and builds are run before commit. No native GPU, trained checkpoint, corpus, private audio, service, deployment, push, dependency pin or performance run is part of this checkpoint.
