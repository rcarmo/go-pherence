#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "model/speculative/drafter"

# Batch 28: mechanically split speculative drafter by stage.
MOVES = {
    "drafter.go": ("core/drafter.go", "core"),
    "drafter_test.go": ("core/drafter_test.go", "core"),
    "loop.go": ("loop/loop.go", "loop"),
    "loop_test.go": ("loop/loop_test.go", "loop"),
    "multi.go": ("multi/multi.go", "multi"),
    "multi_test.go": ("multi/multi_test.go", "multi"),
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

(root / "docs/speculative-drafter-tree-move-table.md").write_text("""# Speculative drafter tree move table

Applied by `scripts/mass_move_project_tree.py` batch 28.

| Concern | Target |
|---|---|
| Drafter core | `model/speculative/drafter/core` |
| Drafter loop | `model/speculative/drafter/loop` |
| Multi-drafter helpers | `model/speculative/drafter/multi` |
""")
