# Community-1 Vulkan embedding rollback ownership — 12 September 2026

Baseline: `c697c7f60bbc110626348e174776ec55f93888cb` on `feat/speech-simd-vulkan`.

## Finding

`NewVulkanResNetTrunk` can return a non-nil stopping owner together with an error when construction fails and its native rollback cannot complete. The caller must retain that owner so `Close` can be retried after drain/quarantine resolution.

`NewVulkanEmbedding` previously called the trunk constructor into local variables and returned `nil, err` immediately. A partial trunk returned with the error was therefore discarded, losing the only retry handle to retained native resources.

## Change

- Added a private trunk-constructor seam to `NewVulkanEmbedding`.
- Adopted any non-nil trunk into embedding state before checking its error.
- Added deferred embedding rollback with joined construction/cleanup errors.
- If rollback also fails, return the stopping embedding owner with the error so callers can retry `Close`.
- Reject nil constructor functions and nil successful results.
- Keep forward admission closed after partial rollback; projection state is released only after the retained trunk closes.

No model graph, shader, tensor layout, dispatch, arithmetic, default, quality threshold or public constructor signature changed.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- Focused `TestVulkanEmbedding*` tests pass, including injected partial-owner cleanup failure and successful retry.
- Full `backends/vulkan`, `models/speaker/community1` and `runtime/speechjob` packages pass.
- Ten shuffled repetitions of Community-1 Vulkan embedding/ResNet/diarization/segmentation/LSTM tests pass.
- `go vet` passes for the three affected packages.
- Linux/arm64 Community-1 test-binary cross-build passes.
- `gofmt` and `git diff --check` pass.

No native GPU, trained checkpoint, corpus, private audio, service, deployment, push, dependency pin or performance run was used.
