# k3bench — per-op backend micro-benchmark suite

Benchmarks each available compute backend tier across the core transformer ops
(**GEMV, RMSNorm, SiLU, RoPE, Attention**), then optionally runs `llama-bench`
on a GGUF model for a whole-model number.

## Usage
`go run ./cmd/k3/k3bench -size 4096 -heads 32 -kv-heads 8 -head-dim 128 -seq 512 -iters 50 -threads 8 [-model m.gguf]`

## go-pherence packages used
- `backends/k3`

## Kernels / SIMD to migrate
- None inline; measures kernels owned by the backend packages. This is the
  benchmark of record for any kernel moved into `backends/spacemit/ime2`,
  `backends/simd/runtime`, or `npu/rvv` — keep it pointed at the shared packages.
