# speccheck — speculative decoding correctness check

Verifies that speculative decoding produces token-identical output to plain
greedy decode (draft acceptance must not change results).

## go-pherence packages used
- `model`, `loader/tokenizer`

## Kernels / SIMD to migrate
- None inline; correctness harness for the `runtime/` speculative path.
