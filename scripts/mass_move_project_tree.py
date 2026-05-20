#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "loader/config"

# Batch 18: mechanically split loader config helpers by model/format concern.
MOVES = {
    "config.go": ("core/config.go", "core"),
    "config_test.go": ("core/config_test.go", "core"),
    "quantization.go": ("quantization/quantization.go", "quantization"),
    "nvfp4_layout.go": ("quantization/nvfp4_layout.go", "quantization"),
    "qwen35_names.go": ("qwen/names.go", "qwen"),
    "qwen35_names_test.go": ("qwen/names_test.go", "qwen"),
    "qwen35_shapes.go": ("qwen/shapes.go", "qwen"),
    "qwen_native_mtp.go": ("qwen/native_mtp.go", "qwen"),
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

(root / "docs/loader-config-tree-move-table.md").write_text("""# Loader config tree move table

Applied by `scripts/mass_move_project_tree.py` batch 18.

| Concern | Target |
|---|---|
| Generic model config parsing/tests | `loader/config/core` |
| Quantization config/layout helpers | `loader/config/quantization` |
| Qwen names/shapes/native-MTP config helpers | `loader/config/qwen` |
""")
