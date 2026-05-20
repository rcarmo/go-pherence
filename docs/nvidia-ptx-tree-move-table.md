# NVIDIA PTX tree move table

Applied by `scripts/mass_move_project_tree.py` batch 17.

| Concern | Target |
|---|---|
| Activation PTX constants | `backends/nvidia/ptx/activation` |
| Attention PTX constants | `backends/nvidia/ptx/attention` |
| Conversion PTX constants | `backends/nvidia/ptx/conversion` |
| LM-head PTX constants | `backends/nvidia/ptx/lmhead` |
| Norm PTX constants | `backends/nvidia/ptx/norm` |
| Prefetch PTX constants | `backends/nvidia/ptx/prefetch` |
| SGEMM PTX constants | `backends/nvidia/ptx/matmul` |
| Vector PTX constants | `backends/nvidia/ptx/vector` |
