#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "tensor/graph"

# Batch 19: mechanically split tensor graph internals by concern.
MOVES = {
    "uop.go": ("core/uop.go", "core"),
    "ops.go": ("core/ops.go", "core"),
    "pattern.go": ("rewrite/pattern.go", "rewrite"),
    "rewrite.go": ("rewrite/rewrite.go", "rewrite"),
    "rewrite_test.go": ("rewrite/rewrite_test.go", "rewrite"),
    "rules.go": ("rewrite/rules.go", "rewrite"),
    "fuse.go": ("opt/fuse.go", "opt"),
    "realize.go": ("runtime/realize.go", "runtime"),
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

(root / "docs/tensor-graph-tree-move-table.md").write_text("""# Tensor graph tree move table

Applied by `scripts/mass_move_project_tree.py` batch 19.

| Concern | Target |
|---|---|
| UOp and graph op definitions | `tensor/graph/core` |
| Pattern rewrite engine/rules/tests | `tensor/graph/rewrite` |
| Graph optimizations/fusion | `tensor/graph/opt` |
| Realization/runtime lowering | `tensor/graph/runtime` |
""")
