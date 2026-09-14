# Community-1 single-cluster K-means validation — 13 September 2026

Baseline: `29e6357ea08074f57c430c17f428b7ba546c270b` on `feat/speech-simd-vulkan`.

## Finding

The forced K-means `clusters == 1` fast path checked each input element for NaN/Inf but returned before the row-wise cosine-normalisation validation used by every multi-cluster path. A zero row or a finite row whose float32 squared norm overflowed could therefore be accepted only when one forced cluster was requested.

## Change

The one-cluster path now computes each row's float32 squared norm, rejects zero or non-finite norms, checks cancellation between bounded row groups, and then returns the deterministic all-zero label vector. It still avoids seeding and Lloyd iterations because those cannot change a one-cluster assignment.

Added regressions for a zero row and a finite `MaxFloat32` overflow row in the one-cluster fixture.

## Verification

With `GO_PHERENCE_DISABLE_NVIDIA=1` and `GOMAXPROCS=2`:

- All 27 pinned K-means oracle cases and validation/cancellation/relocation tests pass.
- Full `models/speaker/community1`, `runtime/speechjob` and `cmd/audio/speechjobserve` tests pass.
- Ten shuffled Community-1 and speech-job repetitions pass.
- Affected `go vet` and native builds pass.
- Linux/arm64 Community-1 test-binary cross-build passes.
- `gofmt` and `git diff --check` pass.

No fixture, processing path, model, GPU, service, default or tolerance changed.
