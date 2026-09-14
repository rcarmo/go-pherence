# Vulkan constructor result contract — 13 September 2026

Baseline: `762690cc05609efbb71ac00a3e481d681759f060` on `feat/speech-simd-vulkan`.

## Finding

Resident Vulkan constructors intentionally permit a non-nil result with an error only when rollback cleanup failed and the stopping owner must remain reachable for `Close` retry. Two Community-1 constructors used `return owner, ctx.Err()` for their final cancellation check. Because their deferred rollback then closed all children successfully, late cancellation could still publish a non-nil, already-closed owner with the error. This contradicted the ownership contract used by nested constructors and server startup.

## Change

- `newVulkanLSTM` and `newVulkanDiarization` now assign the late context error and return nil, allowing deferred rollback to publish an owner only if cleanup itself fails.
- The full constructor/caller scan confirmed that Whisper, basic block, ResNet trunk, embedding, feature segmentation and PCM composition already use nil named results for ordinary failures and adopt partial children before testing errors.
- Server Whisper/Community boundaries already close non-nil error results through the returned owner and close untransferred models directly.
- Added deterministic diarization tests for complete rollback, retained rollback/retry, and late cancellation after both children were constructed.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- Full `backends/vulkan`, `models/whisper`, `models/speaker/community1`, `runtime/speechjob` and `cmd/audio/speechjobserve` tests pass.
- Ten shuffled Community-1 repetitions pass.
- Affected `go vet` and native builds pass.
- Linux/arm64 test-binary cross-builds pass for Vulkan, Whisper and Community-1.
- `gofmt` and `git diff --check` pass.

No native GPU, trained model, corpus, private audio, service or deployment was used.
