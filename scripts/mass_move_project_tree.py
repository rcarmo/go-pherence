#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/quant/q4/ops"

# Batch 25: mechanically split Q4 ops by operation.
MOVES = {
    "dequant.go": ("dequant/dequant.go", "dequant"),
    "gemv.go": ("gemv/gemv.go", "gemv"),
    "gemv_asym_test.go": ("gemv/asym_test.go", "gemv"),
    "gemv_validate_test.go": ("gemv/validate_test.go", "gemv"),
    "gemm.go": ("gemm/gemm.go", "gemm"),
    "gemm_test.go": ("gemm/gemm_test.go", "gemm"),
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

(root / "docs/simd-q4-ops-tree-move-table.md").write_text("""# SIMD Q4 ops tree move table

Applied by `scripts/mass_move_project_tree.py` batch 25.

| Concern | Target |
|---|---|
| Q4 dequant ops | `backends/simd/quant/q4/ops/dequant` |
| Q4 GEMV ops/tests | `backends/simd/quant/q4/ops/gemv` |
| Q4 GEMM ops/tests | `backends/simd/quant/q4/ops/gemm` |
""")
