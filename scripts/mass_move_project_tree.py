#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]

# Batch 3: mechanically split tensor into concern subpackages.
# This only moves files and rewrites package declarations.
MOVES = {
    "tensor/dtype.go": ("tensor/core/dtype.go", "core"),
    "tensor/shape.go": ("tensor/core/shape.go", "core"),
    "tensor/tensor.go": ("tensor/core/tensor.go", "core"),
    "tensor/tensor_test.go": ("tensor/core/tensor_test.go", "core"),
    "tensor/unsafe.go": ("tensor/core/unsafe.go", "core"),
    "tensor/checked.go": ("tensor/core/checked.go", "core"),
    "tensor/broadcast.go": ("tensor/ops/broadcast.go", "ops"),
    "tensor/embedding.go": ("tensor/ops/embedding.go", "ops"),
    "tensor/matmul.go": ("tensor/ops/matmul.go", "ops"),
    "tensor/nn.go": ("tensor/ops/nn.go", "ops"),
    "tensor/modules.go": ("tensor/ops/modules.go", "ops"),
    "tensor/pool.go": ("tensor/ops/pool.go", "ops"),
    "tensor/reference_test.go": ("tensor/ops/reference_test.go", "ops"),
    "tensor/fuse.go": ("tensor/graph/fuse.go", "graph"),
    "tensor/ops.go": ("tensor/graph/ops.go", "graph"),
    "tensor/pattern.go": ("tensor/graph/pattern.go", "graph"),
    "tensor/realize.go": ("tensor/graph/realize.go", "graph"),
    "tensor/rewrite.go": ("tensor/graph/rewrite.go", "graph"),
    "tensor/rewrite_test.go": ("tensor/graph/rewrite_test.go", "graph"),
    "tensor/rules.go": ("tensor/graph/rules.go", "graph"),
    "tensor/uop.go": ("tensor/graph/uop.go", "graph"),
    "tensor/import_boundary_test.go": ("tensor/boundary/import_boundary_test.go", "boundary"),
}

for src_rel, (dst_rel, pkg) in MOVES.items():
    src = root / src_rel
    if not src.exists():
        print(f"skip missing {src_rel}")
        continue
    dst = root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", src_rel, dst_rel], cwd=root, check=True)
    if dst.suffix == ".go":
        text = dst.read_text()
        lines = text.splitlines()
        for i, line in enumerate(lines[:12]):
            if line.startswith("package "):
                lines[i] = f"package {pkg}"
                dst.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
                break

(root / "docs/tensor-tree-move-table.md").write_text("""# Tensor tree move table

Applied by `scripts/mass_move_project_tree.py` batch 3.

| Concern | Target |
|---|---|
| Tensor dtype/shape/core storage/checks | `tensor/core` |
| Tensor math/NN/module operations | `tensor/ops` |
| UOp graph, rewrite, fuse, pattern, rules | `tensor/graph` |
| Import boundary policy tests | `tensor/boundary` |
""")
