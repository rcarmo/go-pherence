#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/quant/q4"

# Batch 12: mechanically split SIMD Q4 quant package by concern.
MOVES = {
    "capabilities.go": ("runtime/capabilities.go", "runtime"),
    "capabilities_test.go": ("runtime/capabilities_test.go", "runtime"),
    "validate.go": ("format/validate.go", "format"),
    "validate_test.go": ("format/validate_test.go", "format"),
    "f16.go": ("format/f16.go", "format"),
    "known_values_test.go": ("format/known_values_test.go", "format"),
    "dequant.go": ("ops/dequant.go", "ops"),
    "gemv.go": ("ops/gemv.go", "ops"),
    "gemv_asym_test.go": ("ops/gemv_asym_test.go", "ops"),
    "gemv_validate_test.go": ("ops/gemv_validate_test.go", "ops"),
    "gemm.go": ("ops/gemm.go", "ops"),
    "gemm_test.go": ("ops/gemm_test.go", "ops"),
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

(root / "docs/simd-q4-tree-move-table.md").write_text("""# SIMD Q4 tree move table

Applied by `scripts/mass_move_project_tree.py` batch 12.

| Concern | Target |
|---|---|
| Q4 capability gates | `backends/simd/quant/q4/runtime` |
| Q4 validation/F16/known values | `backends/simd/quant/q4/format` |
| Q4 dequant/GEMV/GEMM ops | `backends/simd/quant/q4/ops` |
""")
