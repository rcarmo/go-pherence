#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/matmul/sgemm"

# Batch 11: mechanically split SIMD SGEMM variants.
MOVES = {
    "sgemm.go": ("base/sgemm.go", "base"),
    "sgemm_amd64.go": ("base/sgemm_amd64.go", "base"),
    "sgemm_amd64.s": ("base/sgemm_amd64.s", "base"),
    "sgemm_arm64.go": ("base/sgemm_arm64.go", "base"),
    "sgemm_arm64.s": ("base/sgemm_arm64.s", "base"),
    "checked.go": ("checked/checked.go", "checked"),
    "checked_test.go": ("checked/checked_test.go", "checked"),
    "blocked.go": ("blocked/blocked.go", "blocked"),
    "blocked_amd64.go": ("blocked/blocked_amd64.go", "blocked"),
    "blocked_amd64.s": ("blocked/blocked_amd64.s", "blocked"),
    "blocked_arm64.go": ("blocked/blocked_arm64.go", "blocked"),
    "blocked_arm64.s": ("blocked/blocked_arm64.s", "blocked"),
    "blocked_other.go": ("blocked/blocked_other.go", "blocked"),
    "gather.go": ("gather/gather.go", "gather"),
    "gather_amd64.go": ("gather/gather_amd64.go", "gather"),
    "gather_amd64.s": ("gather/gather_amd64.s", "gather"),
    "gather_other.go": ("gather/gather_other.go", "gather"),
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

(root / "docs/simd-sgemm-tree-move-table.md").write_text("""# SIMD SGEMM tree move table

Applied by `scripts/mass_move_project_tree.py` batch 11.

| Concern | Target |
|---|---|
| Base SGEMM assembly/wrappers | `backends/simd/matmul/sgemm/base` |
| Checked SGEMM slice APIs | `backends/simd/matmul/sgemm/checked` |
| Blocked SGEMM variant | `backends/simd/matmul/sgemm/blocked` |
| Gather SGEMM variant | `backends/simd/matmul/sgemm/gather` |
""")
