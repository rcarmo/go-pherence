# shapecheck — tensor shape validator

Cross-checks GGUF tensor shapes against the expected model architecture (head
dims, FFN sizes, expert counts) to catch arch/quant mismatches early.

## go-pherence packages used
- `loader/gguf`

## Kernels / SIMD to migrate
- None inline; validation only.
