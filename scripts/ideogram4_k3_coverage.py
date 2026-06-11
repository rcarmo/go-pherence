#!/usr/bin/env python3
import json
from pathlib import Path

rows = [
  ("FP8 linears", "partial", "RVV fp16 bridge with resident fp16 + N32 packed weights", "final FP8->int8/IME2+TCM"),
  ("FP8 decode", "partial", "row-scaled E4M3 -> resident fp16 bridge", "fused RVV/IME packing"),
  ("Qwen/DiT RMSNorm", "partial", "K3-gated riscv64 seam", "RVV f32/f16 assembly body"),
  ("LayerNorm final", "partial", "K3-gated riscv64 seam", "RVV row LayerNorm assembly body"),
  ("RoPE/MRoPE", "partial", "K3-gated riscv64 seam", "RVV rotate-half assembly body"),
  ("DiT/Qwen attention", "partial", "K3-gated riscv64 seams", "tiled RVV/f16 attention"),
  ("SiLU/Mul/SwiGLU", "partial", "K3-gated riscv64 seams", "RVV vector assembly bodies"),
  ("CFG/update", "partial", "K3-gated riscv64 seam", "RVV vector assembly body"),
  ("VAE Conv2D", "partial", "RVV fp16 im2col GEMM bridge", "resident/tiled RVV or IME conv path"),
  ("VAE GroupNorm/SiLU/Upsample/RGB", "partial", "K3-gated riscv64 seams", "RVV vector assembly bodies"),
  ("VAE spatial attention", "partial", "K3-gated riscv64 seam", "tiled/streaming RVV/f16 attention"),
  ("24GB residency", "partial", "FP8 decoded/packed weight prewarm", "component+activation lifetime policy"),
]
items = [{"area": a, "status": s, "current": c, "remaining": r} for a,s,c,r in rows]
print(json.dumps({"ideogram4_k3_coverage": items}, indent=2))
