#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "backends/simd/kernels"

# Batch 13: mechanically split SIMD scalar kernel bodies by primitive family.
MOVES = {
    "activation.go": ("activation/activation.go", "activation"),
    "activation_test.go": ("activation/activation_test.go", "activation"),
    "gelu.go": ("activation/gelu.go", "activation"),
    "attention.go": ("attention/attention.go", "attention"),
    "layernorm.go": ("norm/layernorm.go", "norm"),
    "softmax.go": ("softmax/softmax.go", "softmax"),
    "rope.go": ("rope/rope.go", "rope"),
    "rope_test.go": ("rope/rope_test.go", "rope"),
    "nn.go": ("nn/nn.go", "nn"),
    "nn_test.go": ("nn/nn_test.go", "nn"),
    "shape.go": ("shape/shape.go", "shape"),
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

(root / "docs/simd-kernels-tree-move-table.md").write_text("""# SIMD kernels tree move table

Applied by `scripts/mass_move_project_tree.py` batch 13.

| Concern | Target |
|---|---|
| Activation/GELU scalar kernels | `backends/simd/kernels/activation` |
| Attention scalar kernels | `backends/simd/kernels/attention` |
| Norm scalar kernels | `backends/simd/kernels/norm` |
| Softmax scalar kernels | `backends/simd/kernels/softmax` |
| RoPE scalar kernels | `backends/simd/kernels/rope` |
| NN helper kernels | `backends/simd/kernels/nn` |
| Shape helpers | `backends/simd/kernels/shape` |
""")
