# embcheck — embedding/tensor sanity check

Inspects embedding (and related) tensors from a GGUF for sanity — dtype, scale,
NaN/Inf — used when debugging tokenizer/embedding mismatches.

## go-pherence packages used
- `loader/gguf`

## Kernels / SIMD to migrate
- None inline; inspection only.
