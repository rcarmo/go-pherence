# Tensor ops tree move table

Applied by `scripts/mass_move_project_tree.py` batch 20.

| Concern | Target |
|---|---|
| Broadcast/shape operation helpers | `tensor/ops/shape` |
| Embedding operations | `tensor/ops/embedding` |
| Matrix multiply operations | `tensor/ops/matmul` |
| NN/pooling operations | `tensor/ops/nn` |
| Module helpers | `tensor/ops/modules` |
| Reference tests | `tensor/ops/reference` |
