#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/matmul/gebp"

# Batch 23: mechanically split SIMD GEBP implementation by provider concern.
MOVES = {
    "gebp.go": ("core/gebp.go", "core"),
    "bounds_test.go": ("core/bounds_test.go", "core"),
    "gebp_amd64.go": ("amd64/gebp.go", "amd64"),
    "gebp_amd64.s": ("amd64/gebp.s", "amd64"),
    "gebp_arm64.go": ("arm64/gebp.go", "arm64"),
    "gebp_arm64.s": ("arm64/gebp.s", "arm64"),
    "gebp_other.go": ("scalar/gebp.go", "scalar"),
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

(root / "docs/simd-gebp-tree-move-table.md").write_text("""# SIMD GEBP tree move table

Applied by `scripts/mass_move_project_tree.py` batch 23.

| Concern | Target |
|---|---|
| GEBP core and bounds tests | `backends/simd/matmul/gebp/core` |
| amd64 GEBP provider | `backends/simd/matmul/gebp/amd64` |
| arm64 GEBP provider | `backends/simd/matmul/gebp/arm64` |
| Portable scalar fallback | `backends/simd/matmul/gebp/scalar` |
""")
