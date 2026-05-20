#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]

# Batch 2: mechanically split the remaining crowded model root into concern
# subpackages. This script only moves files and rewrites package declarations;
# semantic import/API repairs are intentionally left to follow-up passes.
MOVES = {
    # Attention / RoPE
    "model/attention.go": ("model/attention/attention.go", "attention"),
    "model/attention_test.go": ("model/attention/attention_test.go", "attention"),
    "model/rope.go": ("model/rope/rope.go", "rope"),
    "model/rope_test.go": ("model/rope/rope_test.go", "rope"),

    # Linear and dense helpers
    "model/linear_ops.go": ("model/linear/ops.go", "linear"),
    "model/gemv_helpers_test.go": ("model/linear/gemv_helpers_test.go", "linear"),
    "model/bf16.go": ("model/linear/bf16.go", "linear"),

    # MoE
    "model/moe.go": ("model/moe/moe.go", "moe"),
    "model/moe_test.go": ("model/moe/moe_test.go", "moe"),
    "model/moe_compare_test.go": ("model/moe/compare_test.go", "moe"),
    "model/moe_gpu.go": ("model/moe/gpu.go", "moe"),
    "model/moe_gpu_test.go": ("model/moe/gpu_test.go", "moe"),

    # GPU/KV staging
    "model/gpu_forward.go": ("model/gpu/forward.go", "gpu"),
    "model/gpu_kv_copy_test.go": ("model/gpu/kv_copy_test.go", "gpu"),
    "model/kv_staging.go": ("model/kv/staging.go", "kv"),
    "model/kv_staging_test.go": ("model/kv/staging_test.go", "kv"),

    # LM head
    "model/chunked_lm_head.go": ("model/lmhead/chunked.go", "lmhead"),
    "model/chunked_lm_head_test.go": ("model/lmhead/chunked_test.go", "lmhead"),
    "model/lm_head_policy_test.go": ("model/lmhead/policy_test.go", "lmhead"),

    # Decode / prefill / layers
    "model/cpu_decode_step.go": ("model/decode/cpu_step.go", "decode"),
    "model/cpu_decode_step_test.go": ("model/decode/cpu_step_test.go", "decode"),
    "model/batch_prefill.go": ("model/prefill/batch.go", "prefill"),
    "model/batch_prefill_test.go": ("model/prefill/batch_test.go", "prefill"),
    "model/forward_layer.go": ("model/layers/forward.go", "layers"),
    "model/forward_layer_test.go": ("model/layers/forward_test.go", "layers"),

    # Core/shared helpers
    "model/checked.go": ("model/checks/checked.go", "checks"),
    "model/checked_test.go": ("model/checks/checked_test.go", "checks"),
    "model/inference_helpers.go": ("model/inference/helpers.go", "inference"),
    "model/inference_helpers_test.go": ("model/inference/helpers_test.go", "inference"),
    "model/debug.go": ("model/debug/debug.go", "debug"),
    "model/debug_hooks.go": ("model/debug/hooks.go", "debug"),

    # LLaMA shared root types/tests that are no longer root-bound in this cleanup pass
    "model/llama.go": ("model/core/llama.go", "core"),
    "model/llama_test.go": ("model/core/llama_test.go", "core"),
    "model/llama_types.go": ("model/core/types.go", "core"),
    "model/gemma3_generate_test.go": ("model/core/gemma3_generate_test.go", "core"),
    "model/testmain_test.go": ("model/core/testmain_test.go", "core"),

    # Benchmark follows hot primitive helpers rather than staying in root.
    "model/cpu_hotpath_bench_test.go": ("model/bench/cpu_hotpath_bench_test.go", "bench"),
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

(root / "docs/model-tree-move-table.md").write_text("""# Model tree move table

Applied by `scripts/mass_move_project_tree.py` batch 2.

| Concern | Target |
|---|---|
| Attention helpers | `model/attention` |
| RoPE wrappers | `model/rope` |
| Linear/BF16/GEMV helpers | `model/linear` |
| Mixture-of-experts | `model/moe` |
| GPU forward/KV copy hooks | `model/gpu` |
| KV staging | `model/kv` |
| LM-head chunking/policy | `model/lmhead` |
| CPU decode step | `model/decode` |
| Batch prefill | `model/prefill` |
| Layer forward path | `model/layers` |
| Shared checks | `model/checks` |
| Inference helpers | `model/inference` |
| Debug hooks | `model/debug` |
| LLaMA shared core/types | `model/core` |
| CPU hot-path benchmark | `model/bench` |
""")
