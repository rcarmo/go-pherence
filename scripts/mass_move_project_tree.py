#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "tensor/core"

# Batch 26: mechanically split tensor core by storage/type/validation concern.
MOVES = {
    "dtype.go": ("dtype/dtype.go", "dtype"),
    "shape.go": ("shape/shape.go", "shape"),
    "tensor.go": ("storage/tensor.go", "storage"),
    "tensor_test.go": ("storage/tensor_test.go", "storage"),
    "checked.go": ("checks/checked.go", "checks"),
    "unsafe.go": ("unsafe/unsafe.go", "unsafe"),
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

(root / "docs/tensor-core-tree-move-table.md").write_text("""# Tensor core tree move table

Applied by `scripts/mass_move_project_tree.py` batch 26.

| Concern | Target |
|---|---|
| DType definitions | `tensor/core/dtype` |
| Shape helpers | `tensor/core/shape` |
| Tensor storage/tests | `tensor/core/storage` |
| Checked arithmetic/helpers | `tensor/core/checks` |
| Unsafe conversion helpers | `tensor/core/unsafe` |
""")
