# go-264 provider absorption — 13 September 2026

Baseline consumer: `574566d82ee939b80783f09c1024100b7897ba40` on `feat/speech-simd-vulkan`.

Provider: `rcarmo/go-264` commit `48ff0ca8272ad0482565105ef8263762977b873b`, published on `master` as `v0.0.0-20260913104523-48ff0ca8272a`.

## Scope

The existing explicit `loader/audio/media.NewGo264` adapter remains unchanged and FFmpeg remains the speech-job/default backend. This checkpoint updates only the public module pin from `a47077e` to the completed go-264 audio optimisation/documentation tip. No local `replace`, copied provider source, media default, model path, service or deployment changed.

The provider tip includes its completed AAC/WAV/resampler allocation and Plan 9 SIMD work plus final benchmark documentation. It was clean and Rui-authored before publication.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1`, `GOMAXPROCS=2` and `CGO_ENABLED=0`:

- `go mod verify` passes against the public module/checksum service.
- Full `loader/audio/media`, `models/whisper`, `runtime/speechjob` and `cmd/audio/speechjobserve` tests pass.
- Ten shuffled go-264 adapter repetitions pass.
- Affected `go vet` and native builds pass.
- Linux/arm64 and Windows/amd64 media-package test-binary cross-builds pass.
- A fresh external module imports `github.com/rcarmo/go-264/audio`, runs a minimal client, passes `go vet` and verifies its module cache.
- `git diff --check` passes.

Public MINDS/go-264 paired model tests remain separate because the original source WAV fixtures are not currently retained locally. No native model, corpus, service or deployment execution is part of this dependency-only checkpoint.
