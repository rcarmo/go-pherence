# Vulkan cancellation result contract — 13 September 2026

Baseline: `c75c73a61a6d0232e0ed51024365971df8a87323` on `feat/speech-simd-vulkan`.

## Finding

Several resident Vulkan model methods returned an allocated layout or completed-looking inference result together with `ctx.Err()` at their final cancellation checkpoint. This conflicted with the package's checked CPU contract: any error or cancellation returns no partial output. Callers currently discard results on error, but publishing one makes accidental use possible and differs from all earlier cancellation checkpoints.

## Change

- Whisper encoder description and `VulkanEncoder.Forward` now return nil on a final cancellation.
- Community-1 basic-block, ResNet and LSTM graph descriptions now return nil on a final cancellation.
- Community-1 Vulkan basic-block, ResNet, LSTM, embedding, segmentation-PCM and full diarization inference now return no output on a final cancellation.
- Successful non-cancelled results and all arithmetic/layout operations are unchanged.
- Added deterministic final-checkpoint tests for all four model-free graph descriptions and for a fully injected one-window Vulkan diarization run.

A completed native submission remains governed by the existing stage owner: cancellation still triggers fresh-context drain before admission or resource release.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- Focused final-cancellation tests pass.
- Full `backends/vulkan`, `models/whisper`, `models/speaker/community1`, `runtime/speechjob` and `cmd/audio/speechjobserve` tests pass.
- Ten shuffled Whisper and Community-1 repetitions pass.
- Affected `go vet` and native builds pass.
- Linux/arm64 test-binary cross-builds pass for Vulkan, Whisper and Community-1.
- `gofmt` and `git diff --check` pass.

No native GPU, trained model, corpus, private audio, service or deployment was used.
