# Lowered SincNet filters and ordered FMA

Explicit lowered filter weights plus serial-order FMA pass the original `2e-4` final-output gate on all seven pinned synthetic SincNet cases. Scalar Go and Plan 9 AVX2/FMA outputs are bit-exact. One intermediate boundary still exceeds `2e-4`; full SincNet qualification and trained Community-1 integration are unfinished.

## APIs and unchanged defaults

- `NewSincNetWithFilters` accepts an explicit `[80,251]` filter tensor. It validates finiteness, even/odd symmetry, centres and band metadata; copies all weights; and checks cancellation. It does not verify the filters' correspondence to learned bands. The future checkpoint loader must verify provenance, lowering policy and hashes before calling it.
- `SincNetScalarFMA` and `SincNetSIMDFMA` are new explicit modes. Existing `SincNetScalar` and `SincNetSIMD` retain their old reduction order and strict failures. No service or model default changes.
- `FMAColumnsF32Checked` accepts reduction-major `[K,columns]` input, weights `[K]` and output `[columns]`. It rejects bad extents, nonfinite inputs and output overlap before writes. Read-only inputs may overlap. Empty reductions produce positive zeros. Finite overflow is allowed.
- SIMD lanes span independent output frames. Each lane performs the same ascending-K float32 FMA reduction as `FMA32Scalar`; there is no horizontal sum. SincNet packs at most 32 columns and checks cancellation between tiles and channels. Packing is per convolution, not retained model state.
- The assembly uses the existing immutable AVX2/FMA feature probe, checks MXCSR rounding/DAZ/FTZ, restores no controls because it changes none, and issues `VZEROUPPER`. Other architectures use scalar Go. Neither path calls CGo, BLAS, PyTorch or a GPU.

Lowering filters is an offline weight representation. No trained checkpoint exporter or loader is added here. The tests use the existing fixture's explicit filter tensor; it is never compiled into runtime code.

## Diagnostic evidence

The original seven-case fixture regenerates byte-for-byte with the existing pinned exporter: SHA256 `5f2f929595d6ac96e23f4626bca08f5d6fc89311d5913fb193dfe813d52986f1`.

The time vector, Hamming window and cutoff products match the reference exactly. Scalar Go sin/cos differs from the installed Torch path by at most `5.960464477539063e-8` on the inspected angles. This creates up to `1.0132789611816406e-6` filter error. The pinned Torch build uses MKL VML for contiguous sine/cosine operations. No MKL or SLEEF implementation was copied.

Exact filters alone do not solve convolution-order differences. Serial FMA with original generated filters also fails. With exact filters plus serial FMA, the two narrow-band final-output errors are:

| Fixture | Maximum absolute error | Gate |
|---|---:|---|
| wave, stride 10 | 0.00019987300038337708 | passes `2e-4`, very small margin |
| wave, stride 1 | 0.000056743621826171875 | passes `2e-4` |

Impulse, silence, constant and both broadband cases pass the same final-output gate. All seven scalar/SIMD output pairs match bit-for-bit, including signed-zero bits.

The stride-10 wave stage-0 boundary error is `0.00038086622953414917`. The new opt-in `TestSincNetLoweredStrictBoundaries` retains two failing subtests (scalar and SIMD) and 12 passing subtests. The original `TestSincNetStrictNarrowBandOracle` retains all four failures, unchanged. The endpoint pass is not a boundary pass. Ordinary runs explicitly skip these two strict tests plus the optional diagnostic trace test.

Diagnostic convolution comparisons show the largest exact-input FMA mismatches near the reference's final frame tiles. The precise MKL tail algorithm has not been reconstructed. A four-lane tail experiment and float64 reductions did not qualify. Earlier failed checks and diagnostic source are retained.

## Verification

- `make speech-sincnet-fma-check speech-affine-check speech-foundations-check speech-media-integration`: pass, including forced AVX2/FMA-off fallback and FFmpeg tests.
- All `backends/simd/...` plus Community-1: 627 passing test events / 270 top-level passes; three explicit skips as above.
- Thirty shuffled new-kernel/lowered endpoint/ownership repetitions: 390 passing events / 180 top-level passes, zero skips/failures.
- Vulkan offline regression: 421 passing events / 121 top-level passes. It exposed old direct-kernel imports in GELU/RoPE oracle tests; commit `a778437` routes them through the existing runtime API. Production Vulkan code is unchanged.
- Kernel tests cover column counts 0–33 around vector/tile boundaries, reduction lengths through 400, eight alignment offsets, canaries, malformed/nonfinite/alias rejection, no allocation, signed zero/subnormal/overflow, MXCSR rejection and protected guard pages for all three buffers. Guard-page columns cover 1–65.
- SincNet tests cover copied filter/weight ownership, malformed filters, unchanged input, stage cancellation and cancellation within packing/channel tiles.
- Community-1 and Vulkan vet pass. SIMD vet still reports the same three pre-existing `q8dot_amd64.s` ABI warnings; other SIMD vet analysers pass. No new assembly ABI warning.
- Community-1 and SIMD arm64 test binaries cross-build. This is not arm64 execution.
- Whole-tree build and backend compile retain baseline errors. Race compilation is unavailable because `gcc` is absent.
- The assembler listing confirms independent-column `VFMADD231PS`/`VFMADD231SS`. Go objdump misdecodes some VEX bytes on this toolchain; it is not used to assert native instruction decoding. GNU/LLVM objdump is unavailable.

Two delegated reviews timed out without findings. The scoped manual adapter review found no immediate source-level blocker, with merge deferred; it does not constitute an independent review of this kernel.

## Reproduce

```sh
GOMAXPROCS=2 CGO_ENABLED=0 make speech-sincnet-fma-check
GOMAXPROCS=2 CGO_ENABLED=0 go test -count=30 -shuffle=on \
  ./backends/simd/runtime ./models/speaker/community1 \
  -run '^Test(FMAColumns|SincNetLowered(Pinned|Ownership))'
# These retain qualification failures; expected nonzero exit:
GO_PHERENCE_TEST_SINCNET_STRICT=1 CGO_ENABLED=0 go test -count=1 \
  ./models/speaker/community1 \
  -run '^TestSincNet(StrictNarrowBandOracle|LoweredStrictBoundaries)$'
```

The reproduction exporter command, environment, evidence hashes and failed diagnostics are in [evidence.json](evidence.json). No performance measurement was made in this checkpoint. CPU overlap with a separate media ASR run was reported around 09:12 UTC; functional comparisons are retained without isolated timing attribution. LLM and speech services remain inactive. No push, deployment, restart or go-264 merge occurred.
