# testi8i4 — INT8×INT4 kernel correctness harness

Standalone check of the `k3I8I4M1` RVV matmul kernel against a scalar reference.
Carries its **own private copy** of `k3_i8i4_go.s` (duplicated from `ime2run`).

## Usage
`go run ./cmd/k3/testi8i4`

## Kernels / SIMD to migrate
- `k3_i8i4_go.s` (`k3I8I4M1`) → `backends/spacemit/ime2/`.
  Once the canonical kernel lives there, this command should import it instead
  of shipping a duplicate, and become a `_test.go` in that package.

## Status
Throwaway correctness probe; fold into `backends/spacemit/ime2` tests on migration.
