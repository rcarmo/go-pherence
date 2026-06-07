# ime2run — pure-Go IME2 inference prototype (kernel motherlode)

End-to-end greedy decode of a GGUF model in **pure Go + RVV/IME2 assembly**, no
CGo for the hot path (`//go:build cgo && q4kcshim` only gates an optional Q4K C
shim for cross-checking). This is the prototyping ground where almost every K3
kernel was first written, tuned, and benchmarked.

## Usage
```
go run -tags 'cgo q4kcshim' ./cmd/ime2run \
  -model <model.gguf> -prompt "..." -tokens 64 \
  -threads 6 -scalar-threads 2 -trace-ids
```

## go-pherence packages used
- `backends/spacemit/ime2` — IME2 vmadot dispatch
- `backends/spacemit/inference` — forward pass scaffolding
- `backends/spacemit/tcm` — tightly-coupled-memory substrate
- `loader/gguf` — weight loading

## Kernels / SIMD to migrate (15 inline `.s` + Go wrappers)
| File | Symbol | Migration target |
|---|---|---|
| `k3_i8i4_go.s` / `_m4_go.s` / `_c_m1_go.s` / `_zpd_go.s` | `k3I8I4M1`, `k3I8I4M4`, `k3I8I4M1C`, `k3I8I4M1ZPDFused` | `backends/spacemit/ime2/` |
| `k3_i8i8_m4_riscv64.s` / `_native_riscv64.s` / `_native_groups_riscv64.s` | `k3I8I8M4`, `k3I8I8M1`, `k3I8I8M1Groups` | `backends/spacemit/ime2/` |
| `q4k_scaled_kernel_riscv64.s` / `_x4_riscv64.s` | `vmadotQ4KIntLoop1024`, `…1024x4` | `backends/spacemit/ime2/` |
| `int8_group_kernel_*.s`, `int8_group_add_*`, `int8_group_i32_*`, `int8_argmax_*` | `vmadotI8Groups1024*`, `vmadotI8ArgmaxGroups1024` | `backends/spacemit/ime2/` |
| `q8_quant_rvv_riscv64.s` | `quantizeQ8Block32RVV` | `backends/simd/runtime/` |
| `copy_rvv_riscv64.s` | `copyBytesRVV` | `backends/simd/runtime/` |

Non-kernel logic that should also relocate:
- **AI-core handshake** — `ai_thread.go`, `ai_barrier.go`, `ai_pool*.go`
  (the `/proc/set_ai_thread` registration + `spine_barrier` pinning to cores
  8–15) → `backends/spacemit/tcm/` or `npu/`.
- **TCM B-wave double-buffer** — `q4k_tcm_bwave.go`, `tcm_parallel.go`,
  `q4k_pair_barrier.go` → `backends/spacemit/tcm/`.
- **FFN / gate fusion** — `q4k_ffn_fuse.go`, `q4k_gate_fuse.go`,
  `q4k_m4_*` → `backends/spacemit/ime2/` (fused-matmul path).
- **Q4K/Q6K repack + extract** — `q6k_native_repack.go`, `q4k_half.go`,
  `q4k_llama_x32.go` → `backends/ggmlquant/` or `ime2/`.
- **Parallel decode + sampler** — `parallel_decode*.go`, `sampler.go`
  → `runtime/` (decode loop + sampling).

## Status
Reference implementation / benchmark harness. Kernels here are the source of
truth; the shared `backends/spacemit/ime2` package should absorb them so other
commands stop carrying private copies (see `testi8i4`).
