# backends/spacemit

Compute support for the **SpaceMIT K3** SoC (8× X60 RISC-V cores, RVV 1.0 +
IME int8, 3 MB TCM SRAM) as used on the MilkV Jupiter 2.

This tree is the **pure-Go, no-cgo** software stack. It is independent of
`backends/k3`, which is the higher-level ORT/Vulkan compute-backend *dispatch*
layer for the same chip.

## Layout

| Package | Role |
|---|---|
| `ime2/` | Low-level IME int8 GEMM kernels (`vmadot`) + a generic `WorkerPool`. Leaf. |
| `rvv/` | Low-level RVV 1.0 SIMD kernels: int8 GEMM, W4A8, copy, q8 quant. Leaf. |
| `tcm/` | TCM (on-chip SRAM) driver. Uncached for CPU → DMA-staging only. Leaf. |
| `inference/` | Mid-level numeric ops (RMSNorm, Q4_K/INT8 mat-vec) over `ime2`. |
| `aicpu/` | The pure-Go transformer inference engine (decode loop + quant kernels). |
| `aicpu/aipool/` | The engine's TCM-aware worker pool. |
| `aicpu/config/` | `IME2_*` environment feature flags. |
| `aicpu/q4kcshim/` | Optional cgo shim linking llama.cpp for kernel validation. |

## Dependency direction

```
aicpu ──> aipool, config, ime2, rvv, tcm, inference
inference ──> ime2
ime2, rvv, tcm ──> (leaves, no internal deps)
```

No import cycles; the leaf kernel packages never depend on the engine.
