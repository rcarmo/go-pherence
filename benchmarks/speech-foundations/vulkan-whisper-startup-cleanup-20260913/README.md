# Vulkan Whisper startup cleanup — 13 September 2026

Baseline: `b1b28be73f240eeb1cfb78671bda146b92e025fa` on `feat/speech-simd-vulkan`.

## Finding

`whisper.NewVulkanEncoder` can return a non-nil stopping encoder together with an error when construction fails and native rollback cannot complete. Its contract requires a fresh `VulkanDrain` before a later `Close` retry when an accepted submission remains.

The server's `closeVulkanEncoder` helper retried `Close` in a loop without draining. Startup could therefore wait forever after a partial encoder construction or after stage construction failed while the encoder retained submitted work. The Community-1 startup path already used drain-aware cleanup.

## Change

- Added an injected drain function to `vulkanProfileRuntime`; the production runtime uses `vk.VulkanDrain`.
- The Vulkan profile rejects runtimes without a drain implementation.
- `closeVulkanEncoder` now calls the configured fresh-context drain after each failed close, then retries.
- `ErrVulkanDeviceLost` and `ErrVulkanUncertain` keep the process in quarantine instead of releasing startup/resource ownership.
- Both partial encoder-construction and stage-construction failure paths use the configured drain interval.

No model, shader, inference path, profile identity, default, service deployment or quality threshold changed.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- A focused injected test verifies `Close → drain → Close` and the configured poll interval.
- Vulkan profile construction/rejection tests pass, including missing drain rejection.
- Full `cmd/audio/speechjobserve`, `runtime/speechjob` and `models/whisper` tests pass.
- Five shuffled server-package repetitions pass.
- Affected `go vet` and native builds pass.
- `gofmt` and `git diff --check` pass.

No native Vulkan device, trained model, corpus, private audio or service was used.
