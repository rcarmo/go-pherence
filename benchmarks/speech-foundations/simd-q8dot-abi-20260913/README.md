# Q8 dot-product Plan 9 ABI metadata repair — 13 September 2026

The amd64 four-row Q8×F32 dot kernel now passes `go vet` without changing its machine operations or Go call signature.

The assembly already wrote four float32 return values at ABI offsets 56, 60, 64 and 68. Its Go declaration used anonymous returns, so the assembler analyser recognised only the first `ret` symbol and reported the other three offsets as invalid. Naming the returns `r0` through `r3` and using those names in assembly aligns the source annotations with the existing layout.

Verification:

- `go vet ./backends/simd/runtime` passes with no warnings;
- 100 repeats of all `DotI8F32` and `DotI8F32x4` tests pass;
- the complete SIMD, Whisper, speech-job and server affected package tests pass;
- the SIMD package compiles for Linux ARM64.

No arithmetic, dispatch policy, model, service, tolerance or default changed.
