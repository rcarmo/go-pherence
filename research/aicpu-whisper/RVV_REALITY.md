# The "NPU" is RVV — there is no descriptor to reverse-engineer

Static disassembly of `libspacemit_ep.so.2.0.2+rc5` (4.9 MB) settles the question
the whole descriptor-RE effort was chasing.

## Evidence

- **1,052,834 instructions, 0 undecoded, 0 custom/unknown opcodes.** Pure
  rv64gcv. No IME/custom matrix extension, no `.insn`, no `(bad)`.
- The int8 GEMM core is standard **RVV**: `vwmacc.vv` (vector widening integer
  multiply-accumulate, int8x int8 -> int32) with `vsext`/`vzext`/`vwcvt.x.x.v`
  for sign extension, `vle8.v`/`vse8.v` for int8 load/store (13.9k/8.3k uses).
- The compute kernels are **MLIR-generated RVV microkernels** in namespace
  `mlir::speir::spekernel::rvv::` — e.g. `spe_mmt4d_transb` (the int8 matmul,
  transposed-B), `spe_pack`, `spe_im2col`, `spe_quantize`/`spe_dequantize`,
  `spe_dwconv2d_nc`. ("speir" = SpacemIT EIR MLIR dialect.)
- **No** `ai_dma`/`aidma`/`doorbell`/`mmio` symbols anywhere; no MMIO store +
  poll loop. The DMA nodes only stage data DRAM<->TCM.

## What the hardware actually is

The SpaceMIT K3 "NPU" int8 acceleration = the **X60 CPU cores running MLIR-
compiled RVV microkernels**, with **TCM** (3 MiB on-chip SRAM) as a fast
scratchpad and the DMA engine staging tiles DRAM<->TCM. There is no separate
matrix engine with a command/descriptor/ring interface. The earlier "0 per-matmul
syscalls" is simply because RVV compute needs no syscalls — it's CPU code.

## Consequence for a pure-Go port

- There is **nothing to "drive"** — no descriptor, no doorbell. The speed is RVV.
- To match the EP in pure Go you would need **RVV int8 microkernels**. As of
  Go 1.24 the compiler/assembler has **no RVV support**; you'd hand-encode RVV via
  `WORD` directives in Go asm — feasible for a focused GEMM hot path but large and
  fragile. The TCM substrate (`npu/tcm.go`) remains valid as the scratchpad layer.
- Realistic options:
  1. **Ship the RTF 0.90 hybrid** (EP RVV kernels via the C++ runner + pure-Go
     turbo decoder) — works today, transcript byte-correct.
  2. **Hand-write an RVV int8 GEMM** in Go assembly (WORD-encoded) on the TCM
     substrate — the only true pure-Go-no-cgo route to RVV speed.
  3. cgo directly into the EP RVV kernels (clean, not pure-Go).

The descriptor-RE track is closed: it was predicated on an NPU command interface
that does not exist on this part.
