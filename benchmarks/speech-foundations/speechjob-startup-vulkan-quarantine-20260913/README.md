# Speech server startup Vulkan quarantine — 13 September 2026

Baseline: `91adc7f5f6e38e1c42796efd34f150dce6ee30db` on `feat/speech-simd-vulkan`.

## Finding

When a raw Whisper encoder or Community-1 Vulkan model was returned with a construction error, server startup retried `Close` after every drain result except explicit device loss or uncertainty. An unexpected drain error therefore allowed another destruction attempt without proving the retained submission idle. A panic from either `Close` or the drain callback also escaped the cleanup boundary. Runtime stage owners already treat those states as process-quarantine conditions.

## Change

- Centralized raw Vulkan startup cleanup for Whisper and Community-1.
- A successful drain or a bounded context deadline/cancellation permits the existing close retry.
- Device loss, uncertainty, any other drain error, a close panic, or a drain panic invokes the quarantine path and retains ownership until process teardown.
- Production quarantine remains an intentional non-returning hold; injected tests use a returning callback to verify classification without hanging the test process.
- Added coverage for successful drain, plain and joined timeout retry, close panic, drain panic, unexpected drain failure, device loss and uncertainty.

No normal profile lifecycle, model, shader, numerical path, default or public API changed.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- Full `cmd/audio/speechjobserve`, `runtime/speechjob`, `models/whisper`, `models/speaker/community1` and `backends/vulkan` tests pass.
- Ten shuffled affected server startup/profile repetitions pass.
- Affected `go vet` and native builds pass.
- Linux/arm64 test-binary cross-builds pass for Vulkan, Whisper and Community-1.
- `gofmt` and `git diff --check` pass.

No native GPU, trained model, corpus, private audio, service or deployment was used.
