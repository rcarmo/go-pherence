#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "model/gemma4/mlp"

# Batch 10: mechanically split Gemma4 MLP diagnostics.
for path in sorted(base.glob("*.go")):
    name = path.name
    if "pli" in name:
        bucket = "pli"
    elif "projection" in name:
        bucket = "projection"
    elif "down" in name or "gate" in name:
        bucket = "down_gate"
    elif name.startswith("quantized_l"):
        if any(tag in name for tag in ["l0_", "l1_", "l2_", "l3_"]):
            bucket = "quantized_early_layers"
        else:
            bucket = "quantized_later_layers"
    elif name.startswith("layer15_"):
        bucket = "layer15"
    elif "standalone" in name:
        bucket = "standalone"
    else:
        bucket = "cpu"
    dst = base / bucket / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(path.relative_to(root)), str(dst.relative_to(root))], cwd=root, check=True)

(root / "docs/gemma4-mlp-tree-move-table.md").write_text("""# Gemma4 MLP diagnostics move table

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
""")
