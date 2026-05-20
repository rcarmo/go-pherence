# Gemma4 MLP diagnostics move table

Applied by `scripts/mass_move_project_tree.py` batch 10.

| Concern | Target |
|---|---|
| CPU/dequantized MLP probes | `model/gemma4/mlp/cpu` |
| Down/gate probes | `model/gemma4/mlp/down_gate` |
| Layer 15 probes | `model/gemma4/mlp/layer15` |
| PLI probes | `model/gemma4/mlp/pli` |
| Projection probes | `model/gemma4/mlp/projection` |
| Early-layer quantized MLP probes | `model/gemma4/mlp/quantized_early_layers` |
| Later-layer quantized MLP probes | `model/gemma4/mlp/quantized_later_layers` |
| Standalone GEMV probes | `model/gemma4/mlp/standalone` |
