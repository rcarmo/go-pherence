#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "model/speculative/verifier"

# Batch 27: mechanically split speculative verifier by stage.
MOVES = {
    "forward.go": ("forward/forward.go", "forward"),
    "forward_test.go": ("forward/forward_test.go", "forward"),
    "plan.go": ("plan/plan.go", "plan"),
    "plan_test.go": ("plan/plan_test.go", "plan"),
    "verify.go": ("verify/verify.go", "verify"),
    "verify_test.go": ("verify/verify_test.go", "verify"),
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

(root / "docs/speculative-verifier-tree-move-table.md").write_text("""# Speculative verifier tree move table

Applied by `scripts/mass_move_project_tree.py` batch 27.

| Concern | Target |
|---|---|
| Verifier forward path | `model/speculative/verifier/forward` |
| Verifier planning | `model/speculative/verifier/plan` |
| Verification execution | `model/speculative/verifier/verify` |
""")
