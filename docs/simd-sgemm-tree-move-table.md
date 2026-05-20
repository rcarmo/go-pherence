# SIMD SGEMM tree move table

Applied by `scripts/mass_move_project_tree.py` batch 11.

| Concern | Target |
|---|---|
| Base SGEMM assembly/wrappers | `backends/simd/matmul/sgemm/base` |
| Checked SGEMM slice APIs | `backends/simd/matmul/sgemm/checked` |
| Blocked SGEMM variant | `backends/simd/matmul/sgemm/blocked` |
| Gather SGEMM variant | `backends/simd/matmul/sgemm/gather` |
