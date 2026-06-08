# backends/spacemit/rvv

Low-level **RISC-V Vector (RVV 1.0)** SIMD kernels for the SpaceMIT K3 X60 cores.

WORD-encoded assembly (the Go 1.24 toolchain has no RVV assembler). Leaf package —
no dependency on the inference engine.

## Contents

| Files | Purpose |
|---|---|
| `gemm.go`, `dot_riscv64.s` | int8 dot product + `GemmI8` / `GemmI8Threaded` |
| `f16.go`, `f16_riscv64.s`, `ker_f16_m4n16_riscv64.s`, `ker_f16_m4n32_riscv64.s` | FP16/Zvfh dot product + `GemmF16` / `GemmF16Outer` / `GemmF16Outer32` / `GemmF16Threaded` (f16 inputs, f32 accumulation/output) |
| `gemm_quant.go` | Dynamic int8 quantize + dequant epilogue (`QuantizeDynamicU8`, `MatMulIntegerDequant`) |
| `ker_w4_riscv64.s`, `gemm_w4.go` | W4A8 (int4-weight) outer-product kernel — built & benchmarked, but **slower** than int8 (no native int4 MAC) |
| `copy_rvv.go` + `.s` | `CopyBytesRVV` / `CopyTCMBytes` byte-copy |
| `q8_quant_rvv.go` + `.s` | `QuantizeQ8Block32RVV` q8 block quantization |

See `research/npu-whisper` for the RVV-vs-IME and W4A8 measurements. The
FP16 path uses the K3's advertised `zvfh`/`zvfhmin` RVV extensions; RVV/Zvfh
instructions are WORD-encoded with shared `k3_isa.h` macros until the Go
assembler supports these mnemonics. The tiled FP16 kernels avoid the scalar-FP
`vfwmacc.vf` form, which traps on the K3 kernel path, by broadcasting A scalars
as halfword bits (`lhu` + `vmv.v.x`) and using the proven `vfwmacc.vv` form.
That broadcast change moved the encoder-shape M4xN32 kernel from ~155 ms to
~51–56 ms at 1500×1280×1280 (8T), so N32 is now the default tile. Generic Zvfh
FP16 is still an X100-standard-RVV path: an `aipool` probe correctly placed
workers on A100 cores 8–15, but the same standard-Zvfh kernel measured ~194 ms
(~25 GF/s) there, versus ~56 ms (~88 GF/s) on X100. A100 needs a real custom-HP
kernel path rather than naive standard Zvfh.
