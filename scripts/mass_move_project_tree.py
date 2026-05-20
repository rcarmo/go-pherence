#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/nvidia/modules"

# Batch 22: mechanically split NVIDIA module/compiler helpers.
MOVES = {
    "compiler.go": ("compiler/compiler.go", "compiler"),
    "compiler_test.go": ("compiler/compiler_test.go", "compiler"),
    "mega_module.go": ("mega/mega_module.go", "mega"),
    "bindings.go": ("bindings/bindings.go", "bindings"),
    "entries.go": ("entries/entries.go", "entries"),
    "state.go": ("state/state.go", "state"),
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

(root / "docs/nvidia-modules-tree-move-table.md").write_text("""# NVIDIA modules tree move table

Applied by `scripts/mass_move_project_tree.py` batch 22.

| Concern | Target |
|---|---|
| PTX compiler helpers/tests | `backends/nvidia/modules/compiler` |
| Mega-module loader | `backends/nvidia/modules/mega` |
| Module bindings | `backends/nvidia/modules/bindings` |
| Module entry metadata | `backends/nvidia/modules/entries` |
| Module state | `backends/nvidia/modules/state` |
""")
