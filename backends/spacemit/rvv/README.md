# backends/spacemit/rvv

Low-level **RISC-V Vector (RVV 1.0)** SIMD kernels for the SpaceMIT K3 X60 cores.

WORD-encoded assembly (the Go 1.24 toolchain has no RVV assembler). Leaf package —
no dependency on the inference engine.

## Contents

| Files | Purpose |
|---|---|
| `gemm.go`, `dot_riscv64.s` | int8 dot product + `GemmI8` / `GemmI8Threaded` |
| `gemm_quant.go` | Dynamic int8 quantize + dequant epilogue (`QuantizeDynamicU8`, `MatMulIntegerDequant`) |
| `ker_w4_riscv64.s`, `gemm_w4.go` | W4A8 (int4-weight) outer-product kernel — built & benchmarked, but **slower** than int8 (no native int4 MAC) |
| `copy_rvv.go` + `.s` | `CopyBytesRVV` / `CopyTCMBytes` byte-copy |
| `q8_quant_rvv.go` + `.s` | `QuantizeQ8Block32RVV` q8 block quantization |

See `research/npu-whisper` for the RVV-vs-IME and W4A8 measurements.
