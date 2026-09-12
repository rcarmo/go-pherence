# Portable legacy mel tables — 12 September 2026

The legacy padded-FFT Whisper GPU helper called `fft.PrecomputeHannWindow` and `fft.PrecomputeMelFilters` from architecture-neutral model code, but both helpers lived in `mel_fused.go` behind `//go:build amd64`. Linux/ARM64 therefore failed to compile `models/whisper` even though table generation itself uses only portable Go/math operations.

## Change

The two helper bodies moved verbatim into untagged `backends/simd/fft/mel_tables.go`. The amd64-only fused mel kernel and its arithmetic are unchanged. Exact-contract 400-point Whisper features remain separate and unchanged.

A portable test checks deterministic 400-point Hann and 80×257 HTK filter tables, finite/non-negative bounded coefficients and non-empty filter support.

## Verification

- Native `backends/simd/fft` and `models/whisper` tests pass.
- Linux/ARM64 test binaries now compile for both packages, including `models/whisper`, which failed before this change with undefined helper symbols.
- Windows/AMD64 `models/whisper` remains blocked by pre-existing Unix-only transitive packages (`spacemit/ime2`, memory advice, Vulkan and NVIDIA loaders); this move neither causes nor claims to fix those separate build boundaries.
- No model, native execution, service, deployment, default, dependency pin or quality threshold changed.
