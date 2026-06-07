# specbench — speculative decoding benchmark

Benchmarks speculative decoding (acceptance rate, effective tok/s) over the
go-pherence model stack — the ngram/prompt-lookup path that doubled decode on
copy-heavy work in the K3 review.

## go-pherence packages used
- `model`, `loader/tokenizer`

## Kernels / SIMD to migrate
- None inline; speculative logic should consolidate under `runtime/`.
