#!/usr/bin/env python3
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
runtime = root / "backends/simd/runtime"

moves = {
    # activation
    "activation.go": ("backends/simd/activation/activation.go", "activation"),
    "activation_checked_test.go": ("backends/simd/activation/activation_checked_test.go", "activation"),
    "bias.go": ("backends/simd/activation/bias.go", "activation"),
    "bias_test.go": ("backends/simd/activation/bias_test.go", "activation"),
    # attention
    "attention.go": ("backends/simd/attention/attention.go", "attention"),
    "attention_test.go": ("backends/simd/attention/attention_test.go", "attention"),
    # dot/scalar
    "simd_amd64.s": ("backends/simd/dot/dot_amd64.s", "dot"),
    "simd_amd64_dispatch.go": ("backends/simd/dot/dispatch_amd64.go", "dot"),
    "simd_arm64.s": ("backends/simd/dot/dot_arm64.s", "dot"),
    "simd_arm64_dispatch.go": ("backends/simd/dot/dispatch_arm64.go", "dot"),
    "simd_other.go": ("backends/simd/dot/dot_other.go", "dot"),
    "scalar.go": ("backends/simd/scalar/dot.go", "scalar"),
    # matmul
    "gebp.go": ("backends/simd/matmul/gebp.go", "matmul"),
    "gebp_amd64.go": ("backends/simd/matmul/gebp_amd64.go", "matmul"),
    "gebp_amd64.s": ("backends/simd/matmul/gebp_amd64.s", "matmul"),
    "gebp_arm64.go": ("backends/simd/matmul/gebp_arm64.go", "matmul"),
    "gebp_arm64.s": ("backends/simd/matmul/gebp_arm64.s", "matmul"),
    "gebp_bounds_test.go": ("backends/simd/matmul/gebp_bounds_test.go", "matmul"),
    "gebp_other.go": ("backends/simd/matmul/gebp_other.go", "matmul"),
    "gemv.go": ("backends/simd/matmul/gemv.go", "matmul"),
    "gemv_test.go": ("backends/simd/matmul/gemv_test.go", "matmul"),
    "pack_amd64.go": ("backends/simd/matmul/pack_amd64.go", "matmul"),
    "pack_amd64.s": ("backends/simd/matmul/pack_amd64.s", "matmul"),
    "pack_arm64.go": ("backends/simd/matmul/pack_arm64.go", "matmul"),
    "pack_arm64.s": ("backends/simd/matmul/pack_arm64.s", "matmul"),
    "pack_arm64_flag.go": ("backends/simd/matmul/pack_arm64_flag.go", "matmul"),
    "pack_other.go": ("backends/simd/matmul/pack_other.go", "matmul"),
    "sgemm.go": ("backends/simd/matmul/sgemm.go", "matmul"),
    "sgemm_amd64.go": ("backends/simd/matmul/sgemm_amd64.go", "matmul"),
    "sgemm_amd64.s": ("backends/simd/matmul/sgemm_amd64.s", "matmul"),
    "sgemm_arm64.go": ("backends/simd/matmul/sgemm_arm64.go", "matmul"),
    "sgemm_arm64.s": ("backends/simd/matmul/sgemm_arm64.s", "matmul"),
    "sgemm_blocked.go": ("backends/simd/matmul/sgemm_blocked.go", "matmul"),
    "sgemm_blocked_amd64.go": ("backends/simd/matmul/sgemm_blocked_amd64.go", "matmul"),
    "sgemm_blocked_amd64.s": ("backends/simd/matmul/sgemm_blocked_amd64.s", "matmul"),
    "sgemm_blocked_arm64.go": ("backends/simd/matmul/sgemm_blocked_arm64.go", "matmul"),
    "sgemm_blocked_arm64.s": ("backends/simd/matmul/sgemm_blocked_arm64.s", "matmul"),
    "sgemm_blocked_other.go": ("backends/simd/matmul/sgemm_blocked_other.go", "matmul"),
    "sgemm_checked.go": ("backends/simd/matmul/checked.go", "matmul"),
    "sgemm_checked_test.go": ("backends/simd/matmul/checked_test.go", "matmul"),
    "sgemm_gather.go": ("backends/simd/matmul/sgemm_gather.go", "matmul"),
    "sgemm_gather_amd64.go": ("backends/simd/matmul/sgemm_gather_amd64.go", "matmul"),
    "sgemm_gather_amd64.s": ("backends/simd/matmul/sgemm_gather_amd64.s", "matmul"),
    "sgemm_gather_other.go": ("backends/simd/matmul/sgemm_gather_other.go", "matmul"),
    # norm / rope / softmax / vector
    "layernorm.go": ("backends/simd/norm/layernorm.go", "norm"),
    "layernorm_test.go": ("backends/simd/norm/layernorm_test.go", "norm"),
    "rope.go": ("backends/simd/rope/rope.go", "rope"),
    "rope_freqs.go": ("backends/simd/rope/freqs.go", "rope"),
    "rope_freqs_test.go": ("backends/simd/rope/freqs_test.go", "rope"),
    "rope_test.go": ("backends/simd/rope/rope_test.go", "rope"),
    "softmax.go": ("backends/simd/softmax/softmax.go", "softmax"),
    "softmax_test.go": ("backends/simd/softmax/softmax_test.go", "softmax"),
    "vec.go": ("backends/simd/vector/vector.go", "vector"),
    "vec_amd64.go": ("backends/simd/vector/vector_amd64.go", "vector"),
    "vec_amd64.s": ("backends/simd/vector/vector_amd64.s", "vector"),
    "vec_arm64.go": ("backends/simd/vector/vector_arm64.go", "vector"),
    "vec_arm64.s": ("backends/simd/vector/vector_arm64.s", "vector"),
    "vec_checked.go": ("backends/simd/vector/checked.go", "vector"),
    "vec_checked_test.go": ("backends/simd/vector/checked_test.go", "vector"),
    "vec_dispatch_asm.go": ("backends/simd/vector/dispatch_asm.go", "vector"),
    "vec_other.go": ("backends/simd/vector/vector_other.go", "vector"),
    "vec_test.go": ("backends/simd/vector/vector_test.go", "vector"),
}

for src_name, (dst_rel, pkg) in moves.items():
    src = runtime / src_name
    if not src.exists():
        raise SystemExit(f"missing source {src}")
    dst = root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", str(src.relative_to(root)), str(dst.relative_to(root))], cwd=root, check=True)
    if dst.suffix == ".go":
        text = dst.read_text()
        text = text.replace("package simd", f"package {pkg}", 1)
        dst.write_text(text)

# Keep runtime as the public facade package for files not moved.
# Update docs that refer to the old one-folder runtime layout.
replacements = {
    "`backends/simd/runtime` for public SIMD dispatch wrappers and assembly/scalar fallback selection.": "`backends/simd/runtime` for the compatibility/public SIMD facade and capability reporting.",
    "`backends/simd/kernels` for reusable CPU kernel bodies split by inference primitive.": "`backends/simd/{activation,attention,dot,matmul,norm,rope,softmax,vector}` for operation-specific CPU/SIMD implementations.",
    "`backends/simd/runtime` | CPU SIMD dispatch facade and assembly/scalar fallback wrappers.": "`backends/simd/runtime` | Compatibility/public CPU SIMD facade and capability reporting.",
}
for path in list((root / "docs").glob("*.md")) + [root / "README.md"]:
    text = path.read_text()
    new = text
    for old, repl in replacements.items():
        new = new.replace(old, repl)
    if new != text:
        path.write_text(new)
