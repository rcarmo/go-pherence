# Speech server owner-constructor admission — 13 September 2026

Baseline: `f03a63610a17347d1865c4bf36b30155327dfab9` on `feat/speech-simd-vulkan`.

## Finding

The server's injected Whisper and Community stage-owner constructors were assumed to return a non-nil owner on success. A nil owner with a nil error reached `owner.Stage()` and panicked during startup. A non-nil owner returned together with an error also needed an explicit ownership rule: cleanup must run through that owner instead of separately closing the model transferred to it.

## Change

- Whisper and Community profile construction reject nil successful owner results.
- If a constructor returns a non-nil owner with an error, startup closes that owner through the existing retry loop.
- If it returns no owner, startup closes/releases the untransferred encoder or model directly.
- CPU Community-1 owner construction uses the same rule and releases the untransferred Go model on nil success.
- Added injected nil-success and partial-owner-error tests for Whisper, CPU Community-1 and Vulkan Community-1 paths.

No production owner constructor currently returns owner-plus-error. The checks keep private seams and future constructors fail closed without leaks, double close or nil dereference.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- Focused server profile/Community construction tests pass.
- Full `cmd/audio/speechjobserve`, `runtime/speechjob`, `models/whisper` and `models/speaker/community1` tests pass.
- Ten shuffled repetitions of affected server profile tests pass.
- Affected `go vet` and native builds pass.
- `gofmt` and `git diff --check` pass.

No native GPU, trained model, corpus, private audio, service or deployment was used.
