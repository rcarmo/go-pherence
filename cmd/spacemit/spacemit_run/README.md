# spacemit_run — multi-backend GGUF inference driver

Runs LLaMA-family inference from a GGUF file through the k3 multi-backend stack.
Backend selection order: `auto → SpacemiT(IME2) → Vulkan → CPU SIMD`; the first
forward pass is benchmarked per tier, then steady-state tokens are timed.

## Usage
`go run ./cmd/spacemit/spacemit_run -model <m.gguf> -prompt "..." -tokens 64 [-backend auto] [-bench-all]`

## go-pherence packages used
- `backends/k3`, `loader/gguf`, `model`

## Kernels / SIMD to migrate
- None inline; pure driver over `backends/k3`. Keep as the user-facing CLI once
  kernels consolidate under the backend packages.
