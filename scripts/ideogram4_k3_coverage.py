#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

rows = [
  ("FP8 linears", "partial", "RVV fp16 bridge with resident fp16 + N32 packed weights", "final FP8->int8/IME2+TCM"),
  ("FP8 decode", "partial", "row-scaled E4M3 -> resident fp16 bridge", "fused RVV/IME packing"),
  ("Qwen/DiT RMSNorm", "partial", "K3-gated riscv64 path composed from existing RVV Snrm2/VecScale/VecMul primitives", "fused RVV row RMSNorm assembly body"),
  ("LayerNorm final", "partial", "K3-gated riscv64 path uses scalar centering plus RVV Snrm2/VecScale", "fused RVV row LayerNorm assembly body"),
  ("RoPE/MRoPE", "partial", "K3-gated riscv64 path composed from existing RVV VecMul/VecScaleAdd primitives", "fused RVV rotate-half assembly body"),
  ("DiT/Qwen attention", "partial", "K3-gated riscv64 seams", "tiled RVV/f16 attention"),
  ("SiLU/Mul/SwiGLU", "partial", "K3-gated riscv64 path uses existing RVV VecMul/VecSiLUMul for Mul and SiLU*Mul; SiLU seam remains scalar", "dedicated RVV SiLU and fused kernels"),
  ("CFG/update", "partial", "K3-gated riscv64 path using existing RVV VecScaleAdd primitives", "fuse into one RVV assembly kernel to remove temporaries"),
  ("VAE Conv2D", "partial", "RVV fp16 im2col GEMM bridge", "resident/tiled RVV or IME conv path"),
  ("VAE GroupNorm/SiLU/Upsample/RGB", "partial", "K3-gated riscv64 seams", "RVV vector assembly bodies"),
  ("VAE spatial attention", "partial", "K3-gated riscv64 seam", "tiled/streaming RVV/f16 attention"),
  ("24GB residency", "partial", "FP8 decoded/packed weight prewarm", "component+activation lifetime policy"),
]
items = [{"area": a, "status": s, "current": c, "remaining": r} for a,s,c,r in rows]
parser = argparse.ArgumentParser()
parser.add_argument("--fail-missing", action="store_true", help="exit non-zero if any coverage row is missing")
args = parser.parse_args()
print(json.dumps({"ideogram4_k3_coverage": items}, indent=2))
if args.fail_missing and any(i["status"] == "missing" for i in items):
    raise SystemExit(1)
