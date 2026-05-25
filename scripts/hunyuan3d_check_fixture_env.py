#!/usr/bin/env python3
"""Check whether the local environment can generate Hunyuan3D fixtures.

This script is intentionally lightweight: it imports optional Python packages,
checks local source/checkpoint/image paths, and emits a JSON report. It does not
load model weights or download large payloads.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PYTHON_DEPS = {
    "metadata_inventory": [],
    "image_preprocess": ["numpy", "PIL", "cv2"],
    "conditioner": ["yaml", "torch", "safetensors.torch", "hy3dgen"],
    "denoiser_step": ["yaml", "torch", "safetensors.torch", "hy3dgen"],
    "lowstep_latents": ["numpy", "yaml", "torch", "safetensors.torch", "hy3dgen"],
    "mesh": ["numpy", "yaml", "torch", "safetensors.torch", "hy3dgen"],
}

FRIENDLY_NAMES = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "safetensors.torch": "safetensors",
    "hy3dgen": "hy3dgen from --hunyuan3d-src",
}


def check_import(module: str) -> dict[str, Any]:
    try:
        importlib.import_module(module)
    except Exception as exc:
        return {"ok": False, "module": module, "package": FRIENDLY_NAMES.get(module, module), "error": str(exc)}
    return {"ok": True, "module": module, "package": FRIENDLY_NAMES.get(module, module)}


def path_status(path: str, kind: str) -> dict[str, Any]:
    if not path:
        return {"ok": False, "path": path, "kind": kind, "error": "not provided"}
    p = Path(path)
    if kind == "dir":
        ok = p.is_dir()
    else:
        ok = p.is_file()
    out = {"ok": ok, "path": str(p), "kind": kind}
    if ok and p.is_file():
        out["bytes"] = p.stat().st_size
    elif not ok:
        out["error"] = "missing"
    return out


def stage_status(name: str, imports: dict[str, dict[str, Any]], paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deps = PYTHON_DEPS[name]
    dep_results = [imports[d] for d in deps]
    needed_paths = []
    if name in {"conditioner", "denoiser_step", "lowstep_latents", "mesh"}:
        needed_paths = ["hunyuan3d_src", "config", "checkpoint", "image"]
    elif name == "image_preprocess":
        # A synthetic image can be generated if no image is provided.
        needed_paths = []
    path_results = [paths[p] for p in needed_paths]
    ok = all(d["ok"] for d in dep_results) and all(p["ok"] for p in path_results)
    return {"ok": ok, "deps": dep_results, "paths": path_results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hunyuan3d-src", default="/workspace/tmp/Hunyuan3D-2-info")
    parser.add_argument("--config", default="", help="local Hunyuan3D config.yaml")
    parser.add_argument("--checkpoint", default="", help="local Hunyuan3D model checkpoint")
    parser.add_argument("--image", default="", help="local input image; optional for synthetic image preprocessing fixture")
    parser.add_argument("--out", default="", help="optional JSON report path")
    args = parser.parse_args()

    if args.hunyuan3d_src:
        sys.path.insert(0, args.hunyuan3d_src)

    all_deps = sorted({dep for deps in PYTHON_DEPS.values() for dep in deps})
    imports = {dep: check_import(dep) for dep in all_deps}
    paths = {
        "hunyuan3d_src": path_status(args.hunyuan3d_src, "dir"),
        "config": path_status(args.config, "file"),
        "checkpoint": path_status(args.checkpoint, "file"),
        "image": path_status(args.image, "file"),
    }
    stages = {name: stage_status(name, imports, paths) for name in PYTHON_DEPS}
    report = {
        "schema": "go-pherence-hunyuan3d-fixture-env-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "paths": paths,
        "imports": imports,
        "stages": stages,
        "install_hint": "python3 -m pip install numpy pillow opencv-python pyyaml torch torchvision transformers safetensors einops",
        "source_hint": "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /workspace/tmp/Hunyuan3D-2-info",
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out}")
    else:
        print(text, end="")

    # Non-zero only when the first stage that needs actual local weights cannot run.
    return 0 if stages["conditioner"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
