#!/usr/bin/env python3
"""Generate Hunyuan3D conditioner embedding fixture summaries.

This script is intentionally optional/dependency-gated. It expects a local
Hunyuan3D source checkout plus local config/checkpoint files and emits compact
hash/shape/stat summaries of the image conditioner output. It does not write the
full embedding tensor to JSON by default.

Typical setup:

    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /workspace/tmp/Hunyuan3D-2-info
    python3 -m pip install torch torchvision transformers safetensors pyyaml pillow opencv-python numpy einops

Example:

    python3 scripts/hunyuan3d_conditioner_fixture.py \
      --hunyuan3d-src /workspace/tmp/Hunyuan3D-2-info \
      --config models/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini/config.yaml \
      --checkpoint models/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini/model.fp16.safetensors \
      --image assets/demo.png \
      --out /workspace/tmp/hunyuan3d-conditioner-fixture.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def require_deps(hunyuan3d_src: str):
    if hunyuan3d_src:
        sys.path.insert(0, hunyuan3d_src)
    missing = []
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
        missing.append("pyyaml")
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
        missing.append("torch")
    try:
        import safetensors.torch  # type: ignore
    except Exception:
        safetensors_torch = None
        missing.append("safetensors")
    else:
        safetensors_torch = safetensors.torch
    try:
        from hy3dgen.shapegen.pipelines import instantiate_from_config  # type: ignore
    except Exception:
        instantiate_from_config = None
        missing.append("hy3dgen from --hunyuan3d-src")
    if missing:
        print(
            "missing Python conditioner fixture dependencies: "
            + ", ".join(missing)
            + "\ninstall Hunyuan3D deps and pass --hunyuan3d-src to a local Tencent-Hunyuan/Hunyuan3D-2 checkout",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return yaml, torch, safetensors_torch, instantiate_from_config


def tensor_summary(torch, name: str, value) -> dict[str, Any]:
    v = value.detach().to("cpu", dtype=torch.float32).contiguous()
    raw = v.numpy().astype("<f4", copy=False).tobytes()
    flat = v.reshape(-1)
    sample_count = min(16, flat.numel())
    return {
        "name": name,
        "dtype": "float32",
        "shape": list(v.shape),
        "sha256_le_f32": hashlib.sha256(raw).hexdigest(),
        "min": float(v.min().item()) if flat.numel() else 0.0,
        "max": float(v.max().item()) if flat.numel() else 0.0,
        "mean": float(v.mean().item()) if flat.numel() else 0.0,
        "first_values": [float(x) for x in flat[:sample_count].tolist()],
    }


def strip_group(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = {}
    dot = prefix + "."
    for key, value in state.items():
        if key.startswith(dot):
            out[key[len(dot):]] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hunyuan3d-src", default="/workspace/tmp/Hunyuan3D-2-info", help="local Hunyuan3D-2 source checkout")
    parser.add_argument("--config", required=True, help="Hunyuan3D shape config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Hunyuan3D model safetensors/ckpt containing conditioner.* weights")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--out", default="testdata/hunyuan3d/conditioner-fixture.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()

    yaml, torch, safetensors_torch, instantiate_from_config = require_deps(args.hunyuan3d_src)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_processor = instantiate_from_config(cfg["image_processor"])
    conditioner = instantiate_from_config(cfg["conditioner"])

    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix == ".safetensors":
        state = safetensors_torch.load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if "conditioner" in state:
            state = {"conditioner." + k: v for k, v in state["conditioner"].items()}

    conditioner_state = strip_group(state, "conditioner")
    if not conditioner_state:
        raise SystemExit(f"no conditioner.* tensors found in {checkpoint}")
    missing, unexpected = conditioner.load_state_dict(conditioner_state, strict=False)
    conditioner.to(device=args.device, dtype=dtype)
    conditioner.eval()

    cond_inputs = image_processor(args.image)
    image = cond_inputs.pop("image").to(device=args.device, dtype=dtype)
    cond_inputs = {k: v.to(device=args.device, dtype=dtype) if hasattr(v, "to") else v for k, v in cond_inputs.items()}

    with torch.inference_mode():
        cond = conditioner(image=image, **cond_inputs)
        uncond = conditioner.unconditional_embedding(image.shape[0], **cond_inputs)

    outputs = []
    for group_name, group in [("conditioned", cond), ("unconditioned", uncond)]:
        for key, value in sorted(group.items()):
            outputs.append(tensor_summary(torch, f"{group_name}.{key}", value))

    fixture = {
        "schema": "go-pherence-hunyuan3d-conditioner-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "hunyuan3d_src": args.hunyuan3d_src,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "image": args.image,
            "device": args.device,
            "dtype": args.dtype,
        },
        "load_state": {
            "missing": list(missing),
            "unexpected": list(unexpected),
            "conditioner_tensor_count": len(conditioner_state),
        },
        "outputs": outputs,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
