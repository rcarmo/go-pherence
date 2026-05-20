#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "model/speculative"

# Batch 8: mechanically split speculative/MTP implementation into subpackages.
# Moves files and rewrites package declarations only.
MOVES = {
    "drafter.go": ("drafter/drafter.go", "drafter"),
    "drafter_test.go": ("drafter/drafter_test.go", "drafter"),
    "drafter_loop.go": ("drafter/loop.go", "drafter"),
    "drafter_loop_test.go": ("drafter/loop_test.go", "drafter"),
    "drafter_multi.go": ("drafter/multi.go", "drafter"),
    "drafter_multi_test.go": ("drafter/multi_test.go", "drafter"),
    "verifier_forward.go": ("verifier/forward.go", "verifier"),
    "verifier_forward_test.go": ("verifier/forward_test.go", "verifier"),
    "verifier_plan.go": ("verifier/plan.go", "verifier"),
    "verifier_plan_test.go": ("verifier/plan_test.go", "verifier"),
    "verify.go": ("verifier/verify.go", "verifier"),
    "verify_test.go": ("verifier/verify_test.go", "verifier"),
    "mtp_accept.go": ("accept/mtp_accept.go", "accept"),
    "mtp_accept_test.go": ("accept/mtp_accept_test.go", "accept"),
    "step.go": ("step/step.go", "step"),
    "step_test.go": ("step/step_test.go", "step"),
    "state.go": ("state/state.go", "state"),
    "state_test.go": ("state/state_test.go", "state"),
    "stats.go": ("stats/stats.go", "stats"),
    "stats_test.go": ("stats/stats_test.go", "stats"),
    "speculative.go": ("core/speculative.go", "core"),
    "speculative_test.go": ("core/speculative_test.go", "core"),
    "integration_test.go": ("integration/integration_test.go", "integration"),
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

(root / "docs/speculative-tree-move-table.md").write_text("""# Speculative/MTP tree move table

Applied by `scripts/mass_move_project_tree.py` batch 8.

| Concern | Target |
|---|---|
| Drafter implementations | `model/speculative/drafter` |
| Verifier planning/forward/verify | `model/speculative/verifier` |
| Token acceptance | `model/speculative/accept` |
| Speculative step | `model/speculative/step` |
| Speculative state | `model/speculative/state` |
| Speculative stats | `model/speculative/stats` |
| Core speculative helpers | `model/speculative/core` |
| Integration tests | `model/speculative/integration` |
""")
