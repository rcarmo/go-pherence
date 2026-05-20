#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/matmul"

# Batch 7: mechanically split SIMD matmul internals into operation subpackages.
# Moves files and rewrites package declarations only.
MOVES = {
    "checked.go": ("sgemm/checked.go", "sgemm"),
    "checked_test.go": ("sgemm/checked_test.go", "sgemm"),
    "sgemm.go": ("sgemm/sgemm.go", "sgemm"),
    "sgemm_amd64.go": ("sgemm/sgemm_amd64.go", "sgemm"),
    "sgemm_amd64.s": ("sgemm/sgemm_amd64.s", "sgemm"),
    "sgemm_arm64.go": ("sgemm/sgemm_arm64.go", "sgemm"),
    "sgemm_arm64.s": ("sgemm/sgemm_arm64.s", "sgemm"),
    "sgemm_blocked.go": ("sgemm/blocked.go", "sgemm"),
    "sgemm_blocked_amd64.go": ("sgemm/blocked_amd64.go", "sgemm"),
    "sgemm_blocked_amd64.s": ("sgemm/blocked_amd64.s", "sgemm"),
    "sgemm_blocked_arm64.go": ("sgemm/blocked_arm64.go", "sgemm"),
    "sgemm_blocked_arm64.s": ("sgemm/blocked_arm64.s", "sgemm"),
    "sgemm_blocked_other.go": ("sgemm/blocked_other.go", "sgemm"),
    "sgemm_gather.go": ("sgemm/gather.go", "sgemm"),
    "sgemm_gather_amd64.go": ("sgemm/gather_amd64.go", "sgemm"),
    "sgemm_gather_amd64.s": ("sgemm/gather_amd64.s", "sgemm"),
    "sgemm_gather_other.go": ("sgemm/gather_other.go", "sgemm"),
    "gebp.go": ("gebp/gebp.go", "gebp"),
    "gebp_amd64.go": ("gebp/gebp_amd64.go", "gebp"),
    "gebp_amd64.s": ("gebp/gebp_amd64.s", "gebp"),
    "gebp_arm64.go": ("gebp/gebp_arm64.go", "gebp"),
    "gebp_arm64.s": ("gebp/gebp_arm64.s", "gebp"),
    "gebp_bounds_test.go": ("gebp/bounds_test.go", "gebp"),
    "gebp_other.go": ("gebp/gebp_other.go", "gebp"),
    "pack_amd64.go": ("pack/pack_amd64.go", "pack"),
    "pack_amd64.s": ("pack/pack_amd64.s", "pack"),
    "pack_arm64.go": ("pack/pack_arm64.go", "pack"),
    "pack_arm64.s": ("pack/pack_arm64.s", "pack"),
    "pack_arm64_flag.go": ("pack/arm64_flag.go", "pack"),
    "pack_other.go": ("pack/pack_other.go", "pack"),
    "gemv.go": ("gemv/gemv.go", "gemv"),
    "gemv_test.go": ("gemv/gemv_test.go", "gemv"),
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

(root / "docs/simd-matmul-tree-move-table.md").write_text("""# SIMD matmul tree move table

Applied by `scripts/mass_move_project_tree.py` batch 7.

| Concern | Target |
|---|---|
| Checked/public SGEMM wrappers | `backends/simd/matmul/sgemm` |
| SGEMM assembly and blocked/gather variants | `backends/simd/matmul/sgemm` |
| GEBP kernels/tests | `backends/simd/matmul/gebp` |
| Packing kernels/assembly | `backends/simd/matmul/pack` |
| GEMV references/tests | `backends/simd/matmul/gemv` |
""")
