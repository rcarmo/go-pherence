# Vulkan nil-context admission — 13 September 2026

Baseline: `ac652c99301724bc2612701883692a62bee04300` on `feat/speech-simd-vulkan`.

## Finding

The shared `vkAcquire` function called `ctx.Err()` without checking whether `ctx` was nil. Public APIs that route directly through it could panic when passed a nil context. `VulkanDrain` called `context.WithTimeout` before lane admission and had the same panic class.

## Change

- `vkAcquire(nil)` returns a descriptive error before channel or native access.
- `VulkanDrain(nil, budget)` returns a descriptive error before creating a derived context.
- All public backend constructors, transfers, dispatches and plans that use lane admission inherit the same fail-closed behavior.
- Added a shared regression that covers lane admission, `DispatchContext`, and `VulkanDrain`, and verifies that no mocked native call or pending submission is created.

No native state machine, shader, model, default or numerical behavior changed for non-nil contexts.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- Focused nil-context and lifetime tests pass.
- Full `backends/vulkan`, `models/whisper`, `models/speaker/community1`, `runtime/speechjob` and `cmd/audio/speechjobserve` tests pass.
- Ten shuffled backend repetitions pass.
- Affected `go vet` and native builds pass.
- Linux/arm64 test-binary cross-builds pass for Vulkan, Whisper and Community-1.
- `gofmt` and `git diff --check` pass.

No native GPU, trained model, corpus, private audio, service or deployment was used.
