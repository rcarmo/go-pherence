# SIMD vector tree move table

Applied by `scripts/mass_move_project_tree.py` batch 15.

| Concern | Target |
|---|---|
| Core vector wrappers/tests | `backends/simd/vector/core` |
| Checked vector APIs | `backends/simd/vector/checked` |
| Dispatch glue | `backends/simd/vector/dispatch` |
| amd64 assembly/provider | `backends/simd/vector/amd64` |
| arm64 assembly/provider | `backends/simd/vector/arm64` |
| Portable scalar fallback | `backends/simd/vector/scalar` |
