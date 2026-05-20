#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/nvidia/ptx"

# Batch 17: mechanically split root NVIDIA PTX constants by primitive family.
MOVES = {
    "activation.go": ("activation/activation.go", "activation"),
    "attention.go": ("attention/attention.go", "attention"),
    "conversion.go": ("conversion/conversion.go", "conversion"),
    "lm_head.go": ("lmhead/lm_head.go", "lmhead"),
    "norm.go": ("norm/norm.go", "norm"),
    "prefetch.go": ("prefetch/prefetch.go", "prefetch"),
    "sgemm.go": ("matmul/sgemm.go", "matmul"),
    "vector.go": ("vector/vector.go", "vector"),
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

(root / "docs/nvidia-ptx-tree-move-table.md").write_text("""# NVIDIA PTX tree move table

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
""")
