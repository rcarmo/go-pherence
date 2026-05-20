#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/matmul/pack"

# Batch 24: mechanically split SIMD packing providers by architecture.
MOVES = {
    "pack_amd64.go": ("amd64/pack.go", "amd64"),
    "pack_amd64.s": ("amd64/pack.s", "amd64"),
    "pack_arm64.go": ("arm64/pack.go", "arm64"),
    "pack_arm64.s": ("arm64/pack.s", "arm64"),
    "arm64_flag.go": ("arm64/flag.go", "arm64"),
    "pack_other.go": ("scalar/pack.go", "scalar"),
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

(root / "docs/simd-pack-tree-move-table.md").write_text("""# SIMD pack tree move table

Applied by `scripts/mass_move_project_tree.py` batch 24.

| Concern | Target |
|---|---|
| amd64 packing provider | `backends/simd/matmul/pack/amd64` |
| arm64 packing provider/flags | `backends/simd/matmul/pack/arm64` |
| Portable scalar fallback | `backends/simd/matmul/pack/scalar` |
""")
