# Community speech-job owner result boundary — 13 September 2026

Baseline: `bd53d040898b50c0f17f7e6f240dbe5a8bf9f380` on `feat/speech-simd-vulkan`.

## Finding

The checked Community-1 model contract returns no result with an error, but the CPU and Vulkan speech-job owner seams accept injected implementations for lifecycle testing. If such an implementation returned a non-nil result with an error, each owner propagated both values to the durable stage boundary. The stage currently rejects the error before serialization, but carrying a completed-looking result across the ownership boundary made that invariant depend on downstream behavior.

## Change

- The CPU `Community1Owner` returns nil whenever its model callback returns an error.
- The Vulkan `VulkanCommunity1Stage` clears an errored result before fresh-context drain and return.
- Panic poisoning/quarantine, drain classification, returned errors, stage identity and successful results are unchanged.
- Added direct CPU and Vulkan owner regressions using injected result-plus-error callbacks; the Vulkan test also verifies that drain still occurs once.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- Focused CPU/Vulkan owner lifecycle tests pass.
- Full `runtime/speechjob`, `cmd/audio/speechjobserve`, `models/speaker/community1`, `models/whisper` and `backends/vulkan` tests pass.
- Ten shuffled Community owner/stage repetitions pass.
- Affected `go vet` and native builds pass.
- Linux/arm64 `runtime/speechjob` test-binary cross-build passes.
- `gofmt` and `git diff --check` pass.

No native GPU, trained model, corpus, private audio, service or deployment was used.
