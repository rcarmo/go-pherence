# llmserver — OpenAI-compatible inference server

HTTP server exposing an OpenAI-style chat/completions API over the go-pherence
model stack, with KV cache, speculative decoding (ngram/proposer), turbo-quant
and GPU-offload knobs. The go-pherence analogue of `llama-server`.

## Usage
`go run ./cmd/llmserver -model <m.gguf> -listen :8080 -ctx-size 262144 \
  -cache-type-k f16 -cache-type-v f16 -threads 8 [-speculative -speculative-ngram ...]`

## go-pherence packages used
- `model`, `loader/tokenizer`, `runtime/kv`, `backends/nvidia/runtime`

## Kernels / SIMD to migrate
- None inline; serving layer. Speculative/KV logic belongs in `runtime/`.
