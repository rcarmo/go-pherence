#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/matmul/sgemm/blocked"

# Batch 29: mechanically split blocked SGEMM by provider.
MOVES = {
    "blocked.go": ("core/blocked.go", "core"),
    "blocked_amd64.go": ("amd64/blocked.go", "amd64"),
    "blocked_amd64.s": ("amd64/blocked.s", "amd64"),
    "blocked_arm64.go": ("arm64/blocked.go", "arm64"),
    "blocked_arm64.s": ("arm64/blocked.s", "arm64"),
    "blocked_other.go": ("scalar/blocked.go", "scalar"),
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

(root / "docs/simd-blocked-sgemm-tree-move-table.md").write_text("""# SIMD blocked SGEMM tree move table

Applied by `scripts/mass_move_project_tree.py` batch 29.

| Concern | Target |
|---|---|
| Blocked SGEMM core | `backends/simd/matmul/sgemm/blocked/core` |
| amd64 blocked SGEMM provider | `backends/simd/matmul/sgemm/blocked/amd64` |
| arm64 blocked SGEMM provider | `backends/simd/matmul/sgemm/blocked/arm64` |
| Portable scalar fallback | `backends/simd/matmul/sgemm/blocked/scalar` |
""")
