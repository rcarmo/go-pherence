#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "model/qwen/mtp"

# Batch 21: mechanically split Qwen native MTP helpers by source/harness concern.
MOVES = {
    "mtp.go": ("core/mtp.go", "core"),
    "mtp_test.go": ("core/mtp_test.go", "core"),
    "harness_test.go": ("harness/harness_test.go", "harness"),
    "safetensors.go": ("safetensors/safetensors.go", "safetensors"),
    "safetensors_test.go": ("safetensors/safetensors_test.go", "safetensors"),
    "synthetic.go": ("synthetic/synthetic.go", "synthetic"),
    "synthetic_test.go": ("synthetic/synthetic_test.go", "synthetic"),
}

for src_name, (dst_suffix, pkg) in MOVES.items():
    src = base / src_name
    if not src.exists():
        print(f"skip missing {src.relative_to(root)}")
        continue
    dst = base / dst_suffix
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(src.relative_to(root)), str(dst.relative_to(root))], cwd=root, check=True)
    if dst.suffix == ".go":
        text = dst.read_text()
        lines = text.splitlines()
        for i, line in enumerate(lines[:12]):
            if line.startswith("package "):
                lines[i] = f"package {pkg}"
                dst.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
                break

(root / "docs/qwen-mtp-tree-move-table.md").write_text("""# Qwen MTP tree move table

Applied by `scripts/mass_move_project_tree.py` batch 21.

| Concern | Target |
|---|---|
| Native MTP core | `model/qwen/mtp/core` |
| MTP harness tests | `model/qwen/mtp/harness` |
| Safetensors MTP source helpers | `model/qwen/mtp/safetensors` |
| Synthetic MTP fixtures | `model/qwen/mtp/synthetic` |
""")
