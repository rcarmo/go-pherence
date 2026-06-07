# backends/spacemit/k3engine

The **pure-Go (no-cgo) transformer inference engine** for the SpaceMIT K3 SoC
(MilkV Jupiter 2, 8× X60 RISC-V). `Run()` is the entry point used by
`cmd/k3/ime2run`.

> Not to be confused with `backends/k3`, which is the high-level ORT/Vulkan
> compute-backend *dispatch* layer for the same chip.

## What lives here

| Area | Files |
|---|---|
| Driver / decode loop | `main.go` (`Run`, model load, RoPE/SiLU LUTs), `parallel_decode_ai.go` (+ `_v2/_v3` variants), `sampler.go`, `matvec_hybrid.go` |
| Q4_K path | `q4k_ai.go`, `q4k_llama_x32.go`, `q4k_m4_dispatch.go`, `q4k_m4_pooled.go`, `q4k_gate_fuse.go`, `q4k_ffn_fuse.go`, `q4k_mixed_batch.go`, `q4k_half.go`, `q4k_tcm_bwave.go` |
| Q6_K / Q8 | `q6k_native_repack.go`, `q8_m4_quant.go`, `q8x32_native.go`, `q8x32_native_pooled.go` |
| TCM scheduling | `tcm_parallel.go` |

## Sub-packages

- [`aipool/`](aipool) — the engine's TCM-aware worker pool
- [`config/`](config) — `IME2_*` environment feature flags
- `q4kcshim/` — optional cgo shim linking llama.cpp for kernel validation

## Layering

The raw GEMM kernels do **not** live here — they are in `backends/spacemit/ime2`
(IME `vmadot`: i8i8, i8i4, Q4_K, int8-group, argmax) and `backends/spacemit/rvv`
(RVV: int8 GEMM, copy, q8 quant). This package is the quantization + dispatch +
decode layer that calls down into them. See `backends/spacemit/README.md`.
