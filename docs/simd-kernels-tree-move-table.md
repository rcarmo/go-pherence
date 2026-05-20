# SIMD kernels tree move table

Applied by `scripts/mass_move_project_tree.py` batch 13.

| Concern | Target |
|---|---|
| Activation/GELU scalar kernels | `backends/simd/kernels/activation` |
| Attention scalar kernels | `backends/simd/kernels/attention` |
| Norm scalar kernels | `backends/simd/kernels/norm` |
| Softmax scalar kernels | `backends/simd/kernels/softmax` |
| RoPE scalar kernels | `backends/simd/kernels/rope` |
| NN helper kernels | `backends/simd/kernels/nn` |
| Shape helpers | `backends/simd/kernels/shape` |
