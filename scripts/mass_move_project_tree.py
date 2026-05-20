#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "tensor/ops"

# Batch 20: mechanically split tensor operation helpers by operation family.
MOVES = {
    "broadcast.go": ("shape/broadcast.go", "shape"),
    "embedding.go": ("embedding/embedding.go", "embedding"),
    "matmul.go": ("matmul/matmul.go", "matmul"),
    "nn.go": ("nn/nn.go", "nn"),
    "pool.go": ("nn/pool.go", "nn"),
    "modules.go": ("modules/modules.go", "modules"),
    "reference_test.go": ("reference/reference_test.go", "reference"),
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

(root / "docs/tensor-ops-tree-move-table.md").write_text("""# Tensor ops tree move table

Applied by `scripts/mass_move_project_tree.py` batch 20.

| Concern | Target |
|---|---|
| Broadcast/shape operation helpers | `tensor/ops/shape` |
| Embedding operations | `tensor/ops/embedding` |
| Matrix multiply operations | `tensor/ops/matmul` |
| NN/pooling operations | `tensor/ops/nn` |
| Module helpers | `tensor/ops/modules` |
| Reference tests | `tensor/ops/reference` |
""")
