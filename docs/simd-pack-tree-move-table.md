# SIMD pack tree move table

Applied by `scripts/mass_move_project_tree.py` batch 24.

| Concern | Target |
|---|---|
| amd64 packing provider | `backends/simd/matmul/pack/amd64` |
| arm64 packing provider/flags | `backends/simd/matmul/pack/arm64` |
| Portable scalar fallback | `backends/simd/matmul/pack/scalar` |
