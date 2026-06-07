# verifydot — dot-product kernel verification

Validates the IME2 / RVV fused dot-product path (int4×int8 → fp16 accumulate)
against a scalar golden reference to catch repack/layout regressions.

## Usage
`go run ./cmd/k3/verifydot`

## Kernels / SIMD to migrate
- The verification logic belongs next to the kernels in
  `backends/spacemit/ime2/` (or `npu/rvv/` for the `dot_riscv64.s` path) as a
  table-driven test.

## Status
Diagnostic. Convert to a unit test under the kernel package after migration.
