# SIMD GEBP tree move table

Applied by `scripts/mass_move_project_tree.py` batch 23.

| Concern | Target |
|---|---|
| GEBP core and bounds tests | `backends/simd/matmul/gebp/core` |
| amd64 GEBP provider | `backends/simd/matmul/gebp/amd64` |
| arm64 GEBP provider | `backends/simd/matmul/gebp/arm64` |
| Portable scalar fallback | `backends/simd/matmul/gebp/scalar` |
