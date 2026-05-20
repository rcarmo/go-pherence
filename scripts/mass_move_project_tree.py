#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/matmul/sgemm/base"

# Batch 30: mechanically split base SGEMM by provider.
MOVES = {
    "sgemm.go": ("core/sgemm.go", "core"),
    "sgemm_amd64.go": ("amd64/sgemm.go", "amd64"),
    "sgemm_amd64.s": ("amd64/sgemm.s", "amd64"),
    "sgemm_arm64.go": ("arm64/sgemm.go", "arm64"),
    "sgemm_arm64.s": ("arm64/sgemm.s", "arm64"),
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

(root / "docs/simd-base-sgemm-tree-move-table.md").write_text("""# SIMD base SGEMM tree move table

Applied by `scripts/mass_move_project_tree.py` batch 30.

| Concern | Target |
|---|---|
| Base SGEMM core wrapper | `backends/simd/matmul/sgemm/base/core` |
| amd64 base SGEMM provider | `backends/simd/matmul/sgemm/base/amd64` |
| arm64 base SGEMM provider | `backends/simd/matmul/sgemm/base/arm64` |
""")
