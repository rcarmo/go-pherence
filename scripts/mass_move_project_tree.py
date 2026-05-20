#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]

# Mechanical file moves only. Package declarations are rewritten to the target
# directory package name for Go files. This intentionally does not attempt
# semantic API bridging; follow-up compile fixes can be done after the tree is
# physically organized.
MOVES = {
    # Vulkan split
    "backends/vulkan/vulkan.go": ("backends/vulkan/runtime/vulkan.go", "runtime"),
    "backends/vulkan/debug.go": ("backends/vulkan/runtime/debug.go", "runtime"),
    "backends/vulkan/vulkan_test.go": ("backends/vulkan/runtime/vulkan_test.go", "runtime"),
    "backends/vulkan/vulkan_buf.go": ("backends/vulkan/memory/buf.go", "memory"),
    "backends/vulkan/vulkan_buf_guard_test.go": ("backends/vulkan/memory/buf_guard_test.go", "memory"),
    "backends/vulkan/vulkan_dispatch.go": ("backends/vulkan/dispatch/dispatch.go", "dispatch"),
    "backends/vulkan/vulkan_dispatch_test.go": ("backends/vulkan/dispatch/dispatch_test.go", "dispatch"),
    "backends/vulkan/vulkan_kernel_create_test.go": ("backends/vulkan/dispatch/kernel_create_test.go", "dispatch"),
    "backends/vulkan/vulkan_glsl.go": ("backends/vulkan/shaders/glsl.go", "shaders"),
    "backends/vulkan/vulkan_spirv.go": ("backends/vulkan/shaders/spirv.go", "shaders"),
    "backends/vulkan/vulkan_spirv_embedded.go": ("backends/vulkan/shaders/spirv_embedded.go", "shaders"),
    "backends/vulkan/vulkan_spirv_test.go": ("backends/vulkan/shaders/spirv_test.go", "shaders"),
    "backends/vulkan/vulkan_ops.go": ("backends/vulkan/ops/ops.go", "ops"),
    "backends/vulkan/vulkan_wrapper_test.go": ("backends/vulkan/ops/wrapper_test.go", "ops"),
    "backends/vulkan/vulkan_parity_test.go": ("backends/vulkan/ops/parity_test.go", "ops"),
    "backends/vulkan/vulkan_more_parity_test.go": ("backends/vulkan/ops/more_parity_test.go", "ops"),
    "backends/vulkan/vulkan_bf16_parity_test.go": ("backends/vulkan/ops/bf16_parity_test.go", "ops"),

    # NVIDIA runtime split
    "backends/nvidia/runtime/devbuf.go": ("backends/nvidia/memory/devbuf.go", "memory"),
    "backends/nvidia/runtime/devbuf_authority_test.go": ("backends/nvidia/memory/devbuf_authority_test.go", "memory"),
    "backends/nvidia/runtime/devbuf_state_test.go": ("backends/nvidia/memory/devbuf_state_test.go", "memory"),
    "backends/nvidia/runtime/devbuf_test.go": ("backends/nvidia/memory/devbuf_test.go", "memory"),
    "backends/nvidia/runtime/copy_size_test.go": ("backends/nvidia/memory/copy_size_test.go", "memory"),
    "backends/nvidia/runtime/streams.go": ("backends/nvidia/streams/streams.go", "streams"),
    "backends/nvidia/runtime/streams_test.go": ("backends/nvidia/streams/streams_test.go", "streams"),
    "backends/nvidia/runtime/stats_test.go": ("backends/nvidia/streams/stats_test.go", "streams"),
    "backends/nvidia/runtime/compiler.go": ("backends/nvidia/modules/compiler.go", "modules"),
    "backends/nvidia/runtime/compiler_test.go": ("backends/nvidia/modules/compiler_test.go", "modules"),
    "backends/nvidia/runtime/mega_module.go": ("backends/nvidia/modules/mega_module.go", "modules"),
    "backends/nvidia/runtime/module_bindings.go": ("backends/nvidia/modules/bindings.go", "modules"),
    "backends/nvidia/runtime/module_entries.go": ("backends/nvidia/modules/entries.go", "modules"),
    "backends/nvidia/runtime/module_state.go": ("backends/nvidia/modules/state.go", "modules"),
    "backends/nvidia/runtime/launch_dims.go": ("backends/nvidia/launch/dims.go", "launch"),
    "backends/nvidia/runtime/launch_test.go": ("backends/nvidia/launch/launch_test.go", "launch"),
    "backends/nvidia/runtime/bf16.go": ("backends/nvidia/ops/bf16/bf16.go", "bf16"),
    "backends/nvidia/runtime/bf16_native.go": ("backends/nvidia/ops/bf16/native.go", "bf16"),
    "backends/nvidia/runtime/bf16_test.go": ("backends/nvidia/ops/bf16/bf16_test.go", "bf16"),
    "backends/nvidia/runtime/q4.go": ("backends/nvidia/ops/q4/q4.go", "q4"),
    "backends/nvidia/runtime/gemm_q4.go": ("backends/nvidia/ops/q4/gemm.go", "q4"),
    "backends/nvidia/runtime/gemm_q4_test.go": ("backends/nvidia/ops/q4/gemm_test.go", "q4"),
    "backends/nvidia/runtime/gemm_q4_validation_test.go": ("backends/nvidia/ops/q4/gemm_validation_test.go", "q4"),
    "backends/nvidia/runtime/mlx.go": ("backends/nvidia/ops/mlx/mlx.go", "mlx"),
    "backends/nvidia/runtime/mlx_test.go": ("backends/nvidia/ops/mlx/mlx_test.go", "mlx"),
    "backends/nvidia/runtime/nvfp4.go": ("backends/nvidia/ops/nvfp4/nvfp4.go", "nvfp4"),
    "backends/nvidia/runtime/nvfp4_test.go": ("backends/nvidia/ops/nvfp4/nvfp4_test.go", "nvfp4"),
    "backends/nvidia/runtime/lm_head.go": ("backends/nvidia/ops/lmhead/lm_head.go", "lmhead"),
    "backends/nvidia/runtime/chunked_lm_head.go": ("backends/nvidia/ops/lmhead/chunked.go", "lmhead"),
    "backends/nvidia/runtime/chunked_lm_head_test.go": ("backends/nvidia/ops/lmhead/chunked_test.go", "lmhead"),
    "backends/nvidia/runtime/rope_attention.go": ("backends/nvidia/ops/attention/rope_attention.go", "attention"),
    "backends/nvidia/runtime/sgemm.go": ("backends/nvidia/ops/matmul/sgemm.go", "matmul"),
    "backends/nvidia/runtime/sgemm_test.go": ("backends/nvidia/ops/matmul/sgemm_test.go", "matmul"),
    "backends/nvidia/runtime/expert_pool.go": ("backends/nvidia/experts/pool.go", "experts"),
    "backends/nvidia/runtime/expert_pool_test.go": ("backends/nvidia/experts/pool_test.go", "experts"),

    # Model speculative/MTP split
    "model/mtp_accept.go": ("model/speculative/mtp_accept.go", "speculative"),
    "model/mtp_accept_test.go": ("model/speculative/mtp_accept_test.go", "speculative"),
    "model/mtp_drafter.go": ("model/speculative/drafter.go", "speculative"),
    "model/mtp_drafter_test.go": ("model/speculative/drafter_test.go", "speculative"),
    "model/mtp_drafter_loop.go": ("model/speculative/drafter_loop.go", "speculative"),
    "model/mtp_drafter_loop_test.go": ("model/speculative/drafter_loop_test.go", "speculative"),
    "model/mtp_drafter_multi.go": ("model/speculative/drafter_multi.go", "speculative"),
    "model/mtp_drafter_multi_test.go": ("model/speculative/drafter_multi_test.go", "speculative"),
    "model/mtp_speculative_step.go": ("model/speculative/step.go", "speculative"),
    "model/mtp_speculative_step_test.go": ("model/speculative/step_test.go", "speculative"),
    "model/mtp_stats.go": ("model/speculative/stats.go", "speculative"),
    "model/mtp_stats_test.go": ("model/speculative/stats_test.go", "speculative"),
    "model/mtp_verifier_forward.go": ("model/speculative/verifier_forward.go", "speculative"),
    "model/mtp_verifier_forward_test.go": ("model/speculative/verifier_forward_test.go", "speculative"),
    "model/mtp_verifier_plan.go": ("model/speculative/verifier_plan.go", "speculative"),
    "model/mtp_verifier_plan_test.go": ("model/speculative/verifier_plan_test.go", "speculative"),
    "model/mtp_verify.go": ("model/speculative/verify.go", "speculative"),
    "model/mtp_verify_test.go": ("model/speculative/verify_test.go", "speculative"),
    "model/speculative.go": ("model/speculative/speculative.go", "speculative"),
    "model/speculative_test.go": ("model/speculative/speculative_test.go", "speculative"),
    "model/speculative_state.go": ("model/speculative/state.go", "speculative"),
    "model/speculative_state_test.go": ("model/speculative/state_test.go", "speculative"),
    "model/speculative_integration_test.go": ("model/speculative/integration_test.go", "speculative"),
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
        if "package " in text.splitlines()[0:5].__str__():
            lines = text.splitlines()
            for i, line in enumerate(lines[:10]):
                if line.startswith("package "):
                    lines[i] = f"package {pkg}"
                    dst.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
                    break

# Documentation breadcrumbs.
(root / "docs/project-tree-move-table.md").write_text("""# Project tree move table

Applied by `scripts/mass_move_project_tree.py`.

## Vulkan

- Runtime/device files moved to `backends/vulkan/runtime`.
- Buffer files moved to `backends/vulkan/memory`.
- Dispatch/kernel creation files moved to `backends/vulkan/dispatch`.
- GLSL/SPIR-V files moved to `backends/vulkan/shaders`.
- Operation wrappers and parity tests moved to `backends/vulkan/ops`.

## NVIDIA

- Device buffers moved to `backends/nvidia/memory`.
- Streams/stats moved to `backends/nvidia/streams`.
- Compiler/module loading moved to `backends/nvidia/modules`.
- Launch sizing moved to `backends/nvidia/launch`.
- Operation families moved to `backends/nvidia/ops/{bf16,q4,mlx,nvfp4,lmhead,attention,matmul}`.
- Expert pool logic moved to `backends/nvidia/experts`.

## Model

- Speculative/MTP files moved to `model/speculative`.
""")
