# SIMD matmul tree move table

Applied by `scripts/mass_move_project_tree.py` batch 7.

| Concern | Target |
|---|---|
| Checked/public SGEMM wrappers | `backends/simd/matmul/sgemm` |
| SGEMM assembly and blocked/gather variants | `backends/simd/matmul/sgemm` |
| GEBP kernels/tests | `backends/simd/matmul/gebp` |
| Packing kernels/assembly | `backends/simd/matmul/pack` |
| GEMV references/tests | `backends/simd/matmul/gemv` |
