#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/vector"

# Batch 15: mechanically split SIMD vector package by dispatch/provider concern.
MOVES = {
    "vector.go": ("core/vector.go", "core"),
    "vector_test.go": ("core/vector_test.go", "core"),
    "checked.go": ("checked/checked.go", "checked"),
    "checked_test.go": ("checked/checked_test.go", "checked"),
    "dispatch_asm.go": ("dispatch/asm.go", "dispatch"),
    "vector_amd64.go": ("amd64/vector.go", "amd64"),
    "vector_amd64.s": ("amd64/vector.s", "amd64"),
    "vector_arm64.go": ("arm64/vector.go", "arm64"),
    "vector_arm64.s": ("arm64/vector.s", "arm64"),
    "vector_other.go": ("scalar/vector.go", "scalar"),
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

(root / "docs/simd-vector-tree-move-table.md").write_text("""# SIMD vector tree move table

Applied by `scripts/mass_move_project_tree.py` batch 15.

| Concern | Target |
|---|---|
| Core vector wrappers/tests | `backends/simd/vector/core` |
| Checked vector APIs | `backends/simd/vector/checked` |
| Dispatch glue | `backends/simd/vector/dispatch` |
| amd64 assembly/provider | `backends/simd/vector/amd64` |
| arm64 assembly/provider | `backends/simd/vector/arm64` |
| Portable scalar fallback | `backends/simd/vector/scalar` |
""")
