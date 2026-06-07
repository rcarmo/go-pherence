# backends/spacemit/ime2

Low-level **IME** (Integrated Matrix Engine) int8 matrix-multiply kernels for the
SpaceMIT K3 X60 cores, implemented in native RVV `vmadot` assembly (VLEN=1024).

This is the canonical home for **all** SpaceMIT IME GEMM kernels — commands and
the `k3engine` runtime call into here rather than carrying their own copies.

## Kernel families

| Files | Kernels |
|---|---|
| `ime2*.go`, `gemm*.go` | Core IME int8 GEMM (packed / parallel / direct) + RVV epilogues |
| `int8_group_kernel.go`, `int8_group_add_kernel.go`, `int8_group_i32_kernel.go`, `int8_argmax_kernel.go` | int8 group-GEMM (+ add / int32 / argmax) — `VmadotI8Groups*` |
| `k3_i8i4_*.go` + `k3_i8i4_*_go.s` | i8×i4 (int8 act × int4 weight) M1/M4 kernels, residual & ZPD variants — `K3I8I4*` |
| `k3_i8i8_decl.go` + `k3_i8i8_*_riscv64.s` | i8×i8 native GEMM — `K3I8I8M1/M1Groups/M4` |
| `q4k_scaled_kernel*.go` + `*_riscv64.s` | Q4_K tiled int GEMM — `VmadotQ4KIntLoop1024(x4)` |
| `worker_pool.go` | A generic barrier-based `WorkerPool` for fanning GEMM across cores |

## Notes

- WORD-encoded / native `.s` assembly is used because the Go toolchain has no RVV
  assembler. Most kernels must run on an AI-registered worker goroutine pinned to
  cores 8–15 (see `tcm` and `k3engine/aipool`).
- Leaf package: depends only on `rvv`/`tcm` siblings, never on `k3engine`.
