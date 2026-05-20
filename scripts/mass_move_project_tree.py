#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
base = root / "model/gemma4"

# Batch 6: mechanically split Gemma4 diagnostic tests by investigation area.
# Keep package declarations unchanged (these diagnostics intentionally use
# package model plus build tags/linkname compatibility glue).
BUCKETS = {
    "attention": ["attention_", "attention_test.go", "qknorm"],
    "inputnorm": ["inputnorm", "norm_values", "ffn_norm", "modelbuf_inputnorm", "postgenerate_inputnorm"],
    "gpu": ["gpu_", "gpu"],
    "loader": ["loader_", "loadtime", "encode"],
    "generation": ["generate", "first_token", "nochat", "nopli", "quant_first_tok", "quant_gen2"],
    "mlp": ["mlp", "pli", "projection", "standalone_gemv", "down_", "gate"],
    "quantized": ["quantized_", "quant_", "cpu_quantized", "dequantized", "deq_"],
    "sensitivity": ["sensitivity"],
    "trace": ["trace", "optrace", "layerwalk"],
    "isolation": ["isolation", "corruption", "mv_bug", "stage_fault", "syncdebug", "ablation"],
}

explicit = {
    "compat.go": "support/compat.go",
    "doc.go": "support/doc.go",
}

moves = {}
for path in sorted(base.glob("*.go")):
    name = path.name
    if name in explicit:
        moves[name] = explicit[name]
        continue
    chosen = "misc"
    for bucket, needles in BUCKETS.items():
        if any(name.startswith(n) or n in name for n in needles):
            chosen = bucket
            break
    moves[name] = f"{chosen}/{name}"

for src_name, dst_suffix in moves.items():
    src = base / src_name
    dst = base / dst_suffix
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(src.relative_to(root)), str(dst.relative_to(root))], cwd=root, check=True)

(root / "docs/gemma4-tree-move-table.md").write_text("""# Gemma4 diagnostic tree move table

Applied by `scripts/mass_move_project_tree.py` batch 6.

Gemma4 diagnostic files were mechanically grouped under `model/gemma4` by filename patterns while preserving existing package declarations and build tags.

| Concern | Target |
|---|---|
| Attention/QK norm diagnostics | `model/gemma4/attention` |
| Input/norm diagnostics | `model/gemma4/inputnorm` |
| GPU diagnostics | `model/gemma4/gpu` |
| Loader/load-time diagnostics | `model/gemma4/loader` |
| Generation diagnostics | `model/gemma4/generation` |
| MLP/PLI/projection diagnostics | `model/gemma4/mlp` |
| Quantized CPU/GPU diagnostics | `model/gemma4/quantized` |
| Sensitivity probes | `model/gemma4/sensitivity` |
| Trace/optrace/layerwalk probes | `model/gemma4/trace` |
| Isolation/corruption/fault probes | `model/gemma4/isolation` |
| Compatibility/doc support | `model/gemma4/support` |
| Residual uncategorized probes | `model/gemma4/misc` |
""")
