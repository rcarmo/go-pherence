# SIMD blocked SGEMM tree move table

Applied by `scripts/mass_move_project_tree.py` batch 29.

| Concern | Target |
|---|---|
| Blocked SGEMM core | `backends/simd/matmul/sgemm/blocked/core` |
| amd64 blocked SGEMM provider | `backends/simd/matmul/sgemm/blocked/amd64` |
| arm64 blocked SGEMM provider | `backends/simd/matmul/sgemm/blocked/arm64` |
| Portable scalar fallback | `backends/simd/matmul/sgemm/blocked/scalar` |
