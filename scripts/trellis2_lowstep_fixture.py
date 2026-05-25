#!/usr/bin/env python3
"""Dependency-gated TRELLIS.2 low-step fixture scaffold.

This script is intentionally outside normal Go validation. In a proper upstream
TRELLIS.2 Python environment with local checkpoint payloads it runs the first
low-step sampling stages and emits compact tensor summaries. By default it only
checks dependencies/paths and fails clearly when the heavy runtime is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def require_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"missing dependency {name!r}: {exc}") from exc


def tensor_summary(name: str, tensor: Any, max_values: int = 16) -> dict[str, Any]:
    torch = require_import("torch")
    if hasattr(tensor, "detach"):
        arr = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    else:
        arr = torch.as_tensor(tensor, dtype=torch.float32).contiguous()
    raw = arr.numpy().astype("<f4", copy=False).tobytes()
    flat = arr.flatten()
    return {
        "name": name,
        "dtype": "float32",
        "shape": list(arr.shape),
        "sha256_le_f32": hashlib.sha256(raw).hexdigest(),
        "min": float(flat.min().item()) if flat.numel() else None,
        "max": float(flat.max().item()) if flat.numel() else None,
        "mean": float(flat.mean().item()) if flat.numel() else None,
        "first_values": [float(x) for x in flat[:max_values].tolist()],
    }


def setup_trellis2_path(trellis2_src: Path) -> None:
    if not trellis2_src.exists():
        raise SystemExit(f"TRELLIS.2 source path does not exist: {trellis2_src}")
    sys.path.insert(0, str(trellis2_src))


def load_pipeline(model_dir: str, config_file: str, device: str):
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_dir, config_file=config_file)
    pipe.to(device)
    return pipe


def run_fixture(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    setup_trellis2_path(Path(args.trellis2_src))
    torch = require_import("torch")
    Image = require_import("PIL.Image")
    require_import("safetensors")
    require_import("o_voxel")

    pipe = load_pipeline(args.model_dir, args.config_file, args.device)
    image = Image.open(args.image)
    torch.manual_seed(args.seed)

    with torch.no_grad():
        image = pipe.preprocess_image(image)
        cond = pipe.get_cond(image, args.cond_resolution, include_neg_cond=True)
        out: dict[str, Any] = {
            "schema": "go-pherence-trellis2-lowstep-v1",
            "model_dir": args.model_dir,
            "config_file": args.config_file,
            "device": args.device,
            "seed": args.seed,
            "steps": args.steps,
            "cond_resolution": args.cond_resolution,
            "summaries": [],
        }
        for key, value in cond.items():
            out["summaries"].append(tensor_summary(f"cond.{key}", value))

        ss_params = dict(getattr(pipe, "sparse_structure_sampler_params", {}) or {})
        ss_params.update({"steps": args.steps})
        if args.guidance_strength is not None:
            ss_params["guidance_strength"] = args.guidance_strength
        coords = pipe.sample_sparse_structure(cond, args.structure_resolution, num_samples=1, sampler_params=ss_params)
        out["sparse_structure"] = {
            "shape": list(coords.shape),
            "dtype": str(coords.dtype).replace("torch.", ""),
            "sha256_le_i32": hashlib.sha256(coords.detach().to(device="cpu", dtype=torch.int32).contiguous().numpy().astype("<i4", copy=False).tobytes()).hexdigest(),
            "first_values": coords.detach().to(device="cpu", dtype=torch.int32).flatten()[:32].tolist(),
        }

        if args.run_slat:
            flow_name = "shape_slat_flow_model_512" if args.slat_resolution <= 512 else "shape_slat_flow_model_1024"
            flow_model = pipe.models[flow_name]
            slat_params = dict(getattr(pipe, "shape_slat_sampler_params", {}) or {})
            slat_params.update({"steps": args.steps})
            slat = pipe.sample_shape_slat(cond, flow_model, coords, sampler_params=slat_params)
            out["summaries"].append(tensor_summary("shape_slat.feats", slat.feats))
            out["shape_slat"] = {"coords_shape": list(slat.coords.shape), "flow_model": flow_name}

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trellis2-src", default="/workspace/tmp/TRELLIS.2")
    ap.add_argument("--model-dir", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--config-file", default="pipeline.json")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="/workspace/tmp/trellis2-lowstep-fixture.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--cond-resolution", type=int, default=512)
    ap.add_argument("--structure-resolution", type=int, default=64)
    ap.add_argument("--slat-resolution", type=int, default=512)
    ap.add_argument("--guidance-strength", type=float, default=None)
    ap.add_argument("--run-slat", action="store_true", help="also run shape structured-latent flow after sparse structure")
    args = ap.parse_args()

    report = run_fixture(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
