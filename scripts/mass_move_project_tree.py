#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/quant/nvfp4"

# Batch 14: mechanically split SIMD NVFP4 quant package by concern.
MOVES = {
    "capabilities.go": ("runtime/capabilities.go", "runtime"),
    "capabilities_test.go": ("runtime/capabilities_test.go", "runtime"),
    "types.go": ("format/types.go", "format"),
    "validate.go": ("format/validate.go", "format"),
    "helpers.go": ("format/helpers.go", "format"),
    "decode.go": ("decode/decode.go", "decode"),
    "dequant.go": ("ops/dequant.go", "ops"),
    "gemv.go": ("ops/gemv.go", "ops"),
    "gemm.go": ("ops/gemm.go", "ops"),
    "nvfp4_test.go": ("ops/nvfp4_test.go", "ops"),
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

(root / "docs/simd-nvfp4-tree-move-table.md").write_text("""# SIMD NVFP4 tree move table

Applied by `scripts/mass_move_project_tree.py` batch 14.

| Concern | Target |
|---|---|
| NVFP4 capability gates | `backends/simd/quant/nvfp4/runtime` |
| NVFP4 types/validation/helpers | `backends/simd/quant/nvfp4/format` |
| FP4/F8 decode helpers | `backends/simd/quant/nvfp4/decode` |
| NVFP4 dequant/GEMV/GEMM ops | `backends/simd/quant/nvfp4/ops` |
""")
