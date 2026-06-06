# k3graphrun — llamagraph full GGML decode-graph benchmark

Wires go-pherence's GGUF loader to the `gpll_model` executor and runs the full
GGML decode graph via `backends/llamagraph`. Target: match `k3llama`. Includes
the SpaceMIT repack buffer-type activation that closed the IME2 perf gap.
`//go:build ggml && cgo && linux`.

## Usage
`go run -tags 'ggml cgo linux' ./cmd/k3graphrun -model <m.gguf> -prompt "..." -tokens 64 -threads 8`

## go-pherence packages used
- `backends/llamagraph`, `loader/gguf`

## Kernels / SIMD to migrate
- None inline; orchestrates the GGML/llamagraph backend. The repack/buffer-type
  glue belongs in `backends/llamagraph` (or `backends/ggmlgraph`), not the cmd.
