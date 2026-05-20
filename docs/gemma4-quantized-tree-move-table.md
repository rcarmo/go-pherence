# Gemma4 quantized diagnostics move table

Applied by `scripts/mass_move_project_tree.py` batch 9.

| Concern | Target |
|---|---|
| CPU quantized dimensions/MLX probes | `model/gemma4/quantized/cpu` |
| Quantized transition/optrace probes | `model/gemma4/quantized/trace` |
| Early-layer sensitivity probes | `model/gemma4/quantized/sensitivity/early_layers` |
| Later-layer sensitivity probes | `model/gemma4/quantized/sensitivity/later_layers` |
| Global sensitivity probes | `model/gemma4/quantized/sensitivity/global` |
| Residual probes | `model/gemma4/quantized/misc` |
