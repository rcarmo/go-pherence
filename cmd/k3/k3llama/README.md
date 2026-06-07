# k3llama — libllama.so CGo bridge

Thin CGo bridge that drives the vendor `libllama.so` (SpaceMIT llama.cpp fork)
directly, used as the upper-bound reference for go-pherence's own executors
(~12 tok/s decode baseline on the K3). `//go:build llamacpp && cgo`.

## Usage
`go run -tags 'llamacpp cgo' ./cmd/k3llama -model <m.gguf> -prompt "..." -tokens 64 -threads 8 [-trace-ids -trace-logits -verbose]`

## Kernels / SIMD to migrate
- None of ours; all compute is inside the vendor `.so`. Reference baseline only.
