#!/usr/bin/env python3
"""Run Hunyuan3D-2mini shape generation for a seahorse input image.

This script is intentionally dependency-gated. It expects a local Hunyuan3D-2
checkout (or installed hy3dgen package), PyTorch, and the Hunyuan3D Python
runtime dependencies. It writes an untextured GLB by default because the texture
pipeline requires substantially more VRAM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def require_deps(hunyuan3d_src: str):
    if hunyuan3d_src:
        sys.path.insert(0, hunyuan3d_src)
    missing = []
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
        missing.append("torch")
    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None
        missing.append("pillow")
    try:
        from hy3dgen.rembg import BackgroundRemover  # type: ignore
    except Exception:
        BackgroundRemover = None
        # rembg is optional for RGBA images, but importing BackgroundRemover also
        # proves hy3dgen is importable in typical upstream installs.
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore
    except Exception:
        Hunyuan3DDiTFlowMatchingPipeline = None
        missing.append("hy3dgen shapegen from --hunyuan3d-src / pip install -e")
    if missing:
        raise SystemExit(
            "missing Hunyuan3D seahorse demo dependencies: "
            + ", ".join(missing)
            + "\nInstall PyTorch + Hunyuan3D requirements, then run from a venv."
        )
    return torch, Image, BackgroundRemover, Hunyuan3DDiTFlowMatchingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hunyuan3d-src", default="/workspace/tmp/Hunyuan3D-2-info")
    parser.add_argument("--image", default="testdata/hunyuan3d/seahorse_rgba.png")
    parser.add_argument("--out", default="/workspace/tmp/hunyuan3d-seahorse.glb")
    parser.add_argument("--model", default="tencent/Hunyuan3D-2mini")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mini")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch, Image, BackgroundRemover, Pipeline = require_deps(args.hunyuan3d_src)
    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"input image not found: {image_path}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGBA")
    if image.mode == "RGB" and BackgroundRemover is not None:
        image = BackgroundRemover()(image)

    pipeline = Pipeline.from_pretrained(args.model, subfolder=args.subfolder, variant=args.variant)
    mesh = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=torch.manual_seed(args.seed),
        output_type="trimesh",
    )[0]
    mesh.export(str(out_path))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
