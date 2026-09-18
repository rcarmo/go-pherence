# Community-1 Vulkan PCM wrapper lifetime — 12 September 2026

Baseline: `3cb124183a4a088803f178ea4340ec8f86a397a0` on `feat/speech-simd-vulkan`.

## Finding

`VulkanSegmentationPCM` was the only resident Community-1 wrapper whose fields were stored directly in the public handle instead of shared state. Copying the handle duplicated mutable pointers, `ForwardPCM` and `Close` were not serialized at the wrapper boundary, and one copy could retain a stale frontend/feature pointer after another copy closed the native child.

The constructor also handled a partial feature owner manually rather than adopting it into shared rollback state, making this layer inconsistent with the retryable ownership contract used by the lower resident owners.

## Change

- Moved frontend, feature owner, fixed geometry, stats and lifecycle flags into one shared state object.
- Serialized `ForwardPCM` and `Close` through one gate shared by copied handles.
- `Close` permanently stops admission, waits for active forward work, and is idempotent through every copy.
- Construction adopts a non-nil partial feature owner before checking its error. If construction and cleanup both fail, it returns a stopping PCM owner so `Close` can be retried.
- Added nil-constructor/nil-success, partial rollback/retry, copy-after-close and blocked-close tests.
- Kept immutable stats/grid available without touching closed native state.

No model graph, shader, arithmetic, tensor layout, public constructor signature, default or quality threshold changed.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- Focused `TestVulkanSegmentationPCM*` tests pass.
- Full `backends/vulkan`, `models/speaker/community1`, `runtime/speechjob` and `cmd/audio/speechjobserve` tests pass.
- Ten shuffled repetitions of Community-1 Vulkan embedding/ResNet/diarization/segmentation/LSTM tests pass.
- `go vet` passes for the affected packages.
- Affected native builds pass.
- Linux/arm64 Community-1 test-binary cross-build passes.
- `gofmt`, `git diff --check` and temporary-artifact checks pass.

No native GPU, trained checkpoint, corpus, private audio, service, deployment, push, dependency pin or performance run was used.
