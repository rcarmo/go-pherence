#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/nvidia/ioctl"

# Batch 16: mechanically split NVIDIA ioctl experiment package by concern.
MOVES = {
    "ioctl.go": ("core/ioctl.go", "core"),
    "ioctl_test.go": ("core/ioctl_test.go", "core"),
    "debug.go": ("core/debug.go", "core"),
    "helpers.go": ("helpers/helpers.go", "helpers"),
    "helpers_test.go": ("helpers/helpers_test.go", "helpers"),
    "memory.go": ("memory/memory.go", "memory"),
    "memory_test.go": ("memory/memory_test.go", "memory"),
    "gpfifo.go": ("gpfifo/gpfifo.go", "gpfifo"),
    "query_gpfifo_test.go": ("gpfifo/query_gpfifo_test.go", "gpfifo"),
    "query.go": ("query/query.go", "query"),
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

(root / "docs/nvidia-ioctl-tree-move-table.md").write_text("""# NVIDIA ioctl tree move table

Applied by `scripts/mass_move_project_tree.py` batch 16.

| Concern | Target |
|---|---|
| Core ioctl/debug tests | `backends/nvidia/ioctl/core` |
| Checked helper functions/tests | `backends/nvidia/ioctl/helpers` |
| Memory ioctl experiments | `backends/nvidia/ioctl/memory` |
| GPFIFO experiments/tests | `backends/nvidia/ioctl/gpfifo` |
| Query helpers | `backends/nvidia/ioctl/query` |
""")
