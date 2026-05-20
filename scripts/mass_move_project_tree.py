#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]

# Batch 5: mechanically split Qwen model support into concern subpackages.
# Moves files and rewrites package declarations only.
MOVES = {
    "model/qwen/qwen35.go": ("model/qwen/core/qwen35.go", "core"),
    "model/qwen/qwen35_test.go": ("model/qwen/core/qwen35_test.go", "core"),
    "model/qwen/ops.go": ("model/qwen/core/ops.go", "core"),
    "model/qwen/qwen35_bundle.go": ("model/qwen/bundle/bundle.go", "bundle"),
    "model/qwen/qwen35_bundle_test.go": ("model/qwen/bundle/bundle_test.go", "bundle"),
    "model/qwen/qwen35_source.go": ("model/qwen/source/source.go", "source"),
    "model/qwen/qwen35_source_test.go": ("model/qwen/source/source_test.go", "source"),
    "model/qwen/qwen35_load_helpers.go": ("model/qwen/load/helpers.go", "load"),
    "model/qwen/qwen35_validate_helpers.go": ("model/qwen/load/validate_helpers.go", "load"),
    "model/qwen/qwen35_gpu_cache.go": ("model/qwen/gpu/cache.go", "gpu"),
    "model/qwen/qwen35_mlp_gpu.go": ("model/qwen/gpu/mlp.go", "gpu"),
    "model/qwen/qwen35_linear.go": ("model/qwen/linear/linear.go", "linear"),
    "model/qwen/qwen35_quant.go": ("model/qwen/quant/quant.go", "quant"),
    "model/qwen/qwen35_quant_test.go": ("model/qwen/quant/quant_test.go", "quant"),
    "model/qwen/qwen35_rope.go": ("model/qwen/rope/rope.go", "rope"),
    "model/qwen/qwen35_rope_test.go": ("model/qwen/rope/rope_test.go", "rope"),
    "model/qwen/qwen_native_mtp.go": ("model/qwen/mtp/mtp.go", "mtp"),
    "model/qwen/qwen_native_mtp_test.go": ("model/qwen/mtp/mtp_test.go", "mtp"),
    "model/qwen/qwen_native_mtp_harness_test.go": ("model/qwen/mtp/harness_test.go", "mtp"),
    "model/qwen/qwen_native_mtp_safetensors.go": ("model/qwen/mtp/safetensors.go", "mtp"),
    "model/qwen/qwen_native_mtp_safetensors_test.go": ("model/qwen/mtp/safetensors_test.go", "mtp"),
    "model/qwen/qwen_native_mtp_synthetic.go": ("model/qwen/mtp/synthetic.go", "mtp"),
    "model/qwen/qwen_native_mtp_synthetic_test.go": ("model/qwen/mtp/synthetic_test.go", "mtp"),
}

for src_rel, (dst_rel, pkg) in MOVES.items():
    src = root / src_rel
    if not src.exists():
        print(f"skip missing {src_rel}")
        continue
    dst = root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", src_rel, dst_rel], cwd=root, check=True)
    if dst.suffix == ".go":
        text = dst.read_text()
        lines = text.splitlines()
        for i, line in enumerate(lines[:12]):
            if line.startswith("package "):
                lines[i] = f"package {pkg}"
                dst.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
                break

(root / "docs/qwen-tree-move-table.md").write_text("""# Qwen tree move table

Applied by `scripts/mass_move_project_tree.py` batch 5.

| Concern | Target |
|---|---|
| Qwen core model and shared ops | `model/qwen/core` |
| Bundle loading | `model/qwen/bundle` |
| Safetensors/source helpers | `model/qwen/source` |
| Load/validation helpers | `model/qwen/load` |
| GPU cache/MLP helpers | `model/qwen/gpu` |
| Linear helpers | `model/qwen/linear` |
| Quant helpers | `model/qwen/quant` |
| RoPE helpers | `model/qwen/rope` |
| Native MTP support | `model/qwen/mtp` |
""")
