#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "model/gemma4/quantized"

# Batch 9: mechanically split Gemma4 quantized diagnostics by probe family.
MOVES = {}
for path in sorted(base.glob("*.go")):
    name = path.name
    if name in {"quantized_cpu_dims_test.go", "quantized_cpu_mlx_test.go"}:
        dst = f"cpu/{name}"
    elif "transition_trace" in name or "stepwise_optrace" in name or "cpu_quantized_trace" in name:
        dst = f"trace/{name}"
    elif "sensitivity" in name:
        if any(tag in name for tag in ["l0_", "l1_", "l2_", "l3_"]):
            dst = f"sensitivity/early_layers/{name}"
        elif any(tag in name for tag in ["l4_", "l5_", "l6_", "l7_", "l8_", "l14_"]):
            dst = f"sensitivity/later_layers/{name}"
        else:
            dst = f"sensitivity/global/{name}"
    else:
        dst = f"misc/{name}"
    MOVES[name] = dst

for src_name, dst_suffix in MOVES.items():
    src = base / src_name
    dst = base / dst_suffix
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(src.relative_to(root)), str(dst.relative_to(root))], cwd=root, check=True)

(root / "docs/gemma4-quantized-tree-move-table.md").write_text("""# Gemma4 quantized diagnostics move table

Applied by `scripts/mass_move_project_tree.py` batch 9.

| Concern | Target |
|---|---|
| CPU quantized dimensions/MLX probes | `model/gemma4/quantized/cpu` |
| Quantized transition/optrace probes | `model/gemma4/quantized/trace` |
| Early-layer sensitivity probes | `model/gemma4/quantized/sensitivity/early_layers` |
| Later-layer sensitivity probes | `model/gemma4/quantized/sensitivity/later_layers` |
| Global sensitivity probes | `model/gemma4/quantized/sensitivity/global` |
| Residual probes | `model/gemma4/quantized/misc` |
""")
