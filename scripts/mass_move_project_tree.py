#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]

# Batch 4: mechanically split remaining compact-but-mixed backend/runtime areas.
# Moves files and rewrites package declarations only.
MOVES = {
    # MLX backend split
    "backends/mlx/types.go": ("backends/mlx/format/types.go", "format"),
    "backends/mlx/validate.go": ("backends/mlx/format/validate.go", "format"),
    "backends/mlx/f16.go": ("backends/mlx/format/f16.go", "format"),
    "backends/mlx/helpers.go": ("backends/mlx/format/helpers.go", "format"),
    "backends/mlx/helpers_test.go": ("backends/mlx/format/helpers_test.go", "format"),
    "backends/mlx/load.go": ("backends/mlx/loader/load.go", "loader"),
    "backends/mlx/loader_test.go": ("backends/mlx/loader/loader_test.go", "loader"),
    "backends/mlx/dequant.go": ("backends/mlx/ops/dequant.go", "ops"),
    "backends/mlx/gemv.go": ("backends/mlx/ops/gemv.go", "ops"),
    "backends/mlx/gemm.go": ("backends/mlx/ops/gemm.go", "ops"),
    "backends/mlx/gemm_test.go": ("backends/mlx/ops/gemm_test.go", "ops"),
    "backends/mlx/quant_test.go": ("backends/mlx/ops/quant_test.go", "ops"),
    "backends/mlx/capabilities.go": ("backends/mlx/runtime/capabilities.go", "runtime"),
    "backends/mlx/capabilities_test.go": ("backends/mlx/runtime/capabilities_test.go", "runtime"),

    # Runtime quant compatibility split
    "runtime/quant/gptq.go": ("runtime/quant/q4/gptq.go", "q4"),
    "runtime/quant/gptq_validate.go": ("runtime/quant/q4/gptq_validate.go", "q4"),
    "runtime/quant/gemv_q4.go": ("runtime/quant/q4/gemv.go", "q4"),
    "runtime/quant/gemv_q4_validate.go": ("runtime/quant/q4/gemv_validate.go", "q4"),
    "runtime/quant/mlx.go": ("runtime/quant/mlx/mlx.go", "mlx"),
    "runtime/quant/nvfp4.go": ("runtime/quant/nvfp4/nvfp4.go", "nvfp4"),
    "runtime/quant/compat_checked_test.go": ("runtime/quant/compat/checked_test.go", "compat"),
    "runtime/quant/import_boundary_test.go": ("runtime/quant/boundary/import_boundary_test.go", "boundary"),

    # Runtime KV split
    "runtime/kv/cache.go": ("runtime/kv/cache/cache.go", "cache"),
    "runtime/kv/cache_test.go": ("runtime/kv/cache/cache_test.go", "cache"),
    "runtime/kv/staging.go": ("runtime/kv/staging/staging.go", "staging"),
    "runtime/kv/staging_test.go": ("runtime/kv/staging/staging_test.go", "staging"),
    "runtime/kv/turboquant.go": ("runtime/kv/turboquant/turboquant.go", "turboquant"),
    "runtime/kv/turboquant_test.go": ("runtime/kv/turboquant/turboquant_test.go", "turboquant"),

    # BERT split
    "models/bert/bert.go": ("models/bert/core/bert.go", "core"),
    "models/bert/bert_test.go": ("models/bert/core/bert_test.go", "core"),
    "models/bert/checked.go": ("models/bert/core/checked.go", "core"),
    "models/bert/workspace.go": ("models/bert/core/workspace.go", "core"),
    "models/bert/forward_fast.go": ("models/bert/forward/fast.go", "forward"),
    "models/bert/import_boundary_test.go": ("models/bert/boundary/import_boundary_test.go", "boundary"),
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

(root / "docs/runtime-backend-tree-move-table.md").write_text("""# Runtime/backend tree move table

Applied by `scripts/mass_move_project_tree.py` batch 4.

| Concern | Target |
|---|---|
| MLX format/types/validation/F16 helpers | `backends/mlx/format` |
| MLX loading | `backends/mlx/loader` |
| MLX dequant/GEMV/GEMM ops | `backends/mlx/ops` |
| MLX runtime capabilities | `backends/mlx/runtime` |
| Quant compatibility Q4/GPTQ wrappers | `runtime/quant/q4` |
| Quant compatibility MLX wrappers | `runtime/quant/mlx` |
| Quant compatibility NVFP4 wrappers | `runtime/quant/nvfp4` |
| Quant compatibility/boundary tests | `runtime/quant/{compat,boundary}` |
| KV cache | `runtime/kv/cache` |
| KV staging | `runtime/kv/staging` |
| KV TurboQuant | `runtime/kv/turboquant` |
| BERT core/workspace/checks | `models/bert/core` |
| BERT fast forward | `models/bert/forward` |
| BERT boundary tests | `models/bert/boundary` |
""")
