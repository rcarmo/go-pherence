# Reusable-component consolidation notes

Tracks the consolidation of duplicated reusable primitives across the tree, applying
the same organizational principles used for the `backends/spacemit` kernel filing.

## Guiding rule (layering)

- **Application layer** (`model`, `models`, `loader`, `cmd`) routes shared numeric
  primitives through dedicated leaf packages.
- **Backend packages** (`backends/mlx`, `backends/spacemit`, `backends/simd/quant/*`)
  stay self-contained — they own their codec primitives so each backend remains
  independently portable. Backends are **not** routed through application-layer
  helper packages. Dedup *within* a backend family is fine (see E4M3 below).
- `runtime/quant` is a thin delegation/compat layer over `backends/simd/quant`.

## Half-precision conversion → `half`

The leaf package `half` (`F16ToF32`, `BF16ToF32`; stdlib-only, zero cycle risk)
consolidates conversions that were independently reimplemented in:

- `loader/gguf/dequant.go`, `loader/safetensors` (`float16ToFloat32`),
  `model/moe.go`, `model/ideogram4/fp8_load.go`, `model/gguf_quant_rvv.go`
  (`f16bitsToF32`) — five FP16 decoders.
- `model/moe`, `model/qwen`, `model/ideogram4`, `cmd/qwen/qwen36run` — BF16 decoders.

The five FP16 implementations used different code (bitwise vs `math.Ldexp`) but were
**proven bit-identical for every finite/inf/zero value across all 65536 inputs** (only
NaN bit-payloads differ, which is harmless and never occurs in valid weights). BF16 is
trivially `Float32frombits(uint32(bits) << 16)` everywhere. Consolidation is therefore
behavior-preserving.

Backend f16 decoders (`backends/mlx`, `backends/simd/quant/q4.Float16ToFloat32`, the
cgo C path) and the spacemit `f32ToF16Bits` encoder are intentionally left independent.

## E4M3 scale decode → `backends/simd/quant/fp8`

`backends/simd/quant/nvfp4.DecodeF8E4M3` reimplemented E4M3 decode that was
**bit-identical (verified over all 256 codes)** to `fp8.DecodeE4M3`. Since NVFP4 uses
E4M3 as its per-block scale format, `nvfp4` now delegates to the `fp8` LUT-backed
decoder — a within-family backend dedup that also yields the branch-free table lookup.

## Verified non-targets (left as-is, with reason)

| Candidate | Why left local |
|---|---|
| `sigmoid`, `argmaxF32`, inspection predicates | Trivial helpers; Go idiom "a little copying is better than a little dependency" (and the inspection predicates would cost ~27 call-site changes for ~12 lines) |
| `conv1dForward`, `softmaxInPlace` | Coincidental name collisions — different algorithms (`*1/sum` vs `/sum`, struct+dilation vs flat stride/pad); merging would change outputs |
| Whisper/speaker/loader mel frontends | Different feature pipelines (log-mel vs SpeechBrain fbank); whisper output is byte-exact-constrained |
| lfm2/qwen3tts inspection layer | Generic over package-local types whose JSON shapes feed inspect-command fixtures; a generics refactor would change marshaled output |
| Image/3D model kernels | Architecture-specific (VAE attention, GQA, sparse linear, patch-embed) |

The genuinely reusable low-level kernels (FFT, mel, conv, RMSNorm, gemv, RoPE,
int8/IME GEMM, quant codecs) already live in `backends/simd` and `backends/spacemit`.
