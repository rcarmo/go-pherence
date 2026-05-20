# Gemma4 diagnostic tree move table

Applied by `scripts/mass_move_project_tree.py` batch 6.

Gemma4 diagnostic files were mechanically grouped under `model/gemma4` by filename patterns while preserving existing package declarations and build tags.

| Concern | Target |
|---|---|
| Attention/QK norm diagnostics | `model/gemma4/attention` |
| Input/norm diagnostics | `model/gemma4/inputnorm` |
| GPU diagnostics | `model/gemma4/gpu` |
| Loader/load-time diagnostics | `model/gemma4/loader` |
| Generation diagnostics | `model/gemma4/generation` |
| MLP/PLI/projection diagnostics | `model/gemma4/mlp` |
| Quantized CPU/GPU diagnostics | `model/gemma4/quantized` |
| Sensitivity probes | `model/gemma4/sensitivity` |
| Trace/optrace/layerwalk probes | `model/gemma4/trace` |
| Isolation/corruption/fault probes | `model/gemma4/isolation` |
| Compatibility/doc support | `model/gemma4/support` |
| Residual uncategorized probes | `model/gemma4/misc` |
