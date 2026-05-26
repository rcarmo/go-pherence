#!/usr/bin/env python3
"""Check readiness for TRELLIS.2 dependency-gated fixture generation.

This checker is intentionally lightweight: it imports optional modules when
available, checks local paths, and never loads model weights or runs inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_IMPORTS = [
    "torch",
    "PIL",
    "safetensors",
]
OPTIONAL_RUNTIME_IMPORTS = [
    "flash_attn",
    "xformers",
    "spconv",
    "torchsparse",
    "flex_gemm",
    "nvdiffrast",
]
TRELLIS2_IMPORTS = [
    "trellis2",
    "o_voxel",
]
REQUIRED_MODEL_FILES = [
    "pipeline.json",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
]


def import_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {"name": name, "available": spec is not None, "origin": getattr(spec, "origin", None) if spec else None}


def path_status(path: Path, kind: str) -> dict[str, Any]:
    return {"path": str(path), "kind": kind, "exists": path.exists(), "is_dir": path.is_dir(), "is_file": path.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trellis2-src", default="/workspace/tmp/TRELLIS.2")
    ap.add_argument("--model-dir", default="microsoft/TRELLIS.2-4B", help="local model snapshot path or HF repo id")
    ap.add_argument("--image", default="")
    ap.add_argument("--out", default="/workspace/tmp/trellis2-fixture-env.json")
    args = ap.parse_args()

    src = Path(args.trellis2_src)
    if src.exists():
        sys.path.insert(0, str(src))

    model_is_local = "/" in args.model_dir and Path(args.model_dir).exists() or Path(args.model_dir).exists()
    model_dir = Path(args.model_dir)

    report: dict[str, Any] = {
        "schema": "go-pherence-trellis2-fixture-env-v1",
        "note": "readiness check only; does not load weights or run inference",
        "trellis2_src": path_status(src, "directory"),
        "model_dir": {"value": args.model_dir, "is_local_path": bool(model_is_local)},
        "image": path_status(Path(args.image), "file") if args.image else {"path": "", "exists": False},
        "imports": [import_status(x) for x in REQUIRED_IMPORTS],
        "trellis2_imports": [import_status(x) for x in TRELLIS2_IMPORTS],
        "optional_runtime_imports": [import_status(x) for x in OPTIONAL_RUNTIME_IMPORTS],
        "local_model_files": [],
        "ready_for_lowstep_fixture": False,
        "missing": [],
    }

    for item in REQUIRED_MODEL_FILES:
        if model_is_local:
            report["local_model_files"].append(path_status(model_dir / item, "file"))

    missing: list[str] = []
    if not src.exists():
        missing.append("TRELLIS.2 source checkout")
    for st in report["imports"]:
        if not st["available"]:
            missing.append(f"python import {st['name']}")
    for st in report["trellis2_imports"]:
        if not st["available"]:
            missing.append(f"python import {st['name']} from TRELLIS.2 source/env")
    if args.image and not Path(args.image).exists():
        missing.append("input image")
    if model_is_local:
        for st in report["local_model_files"]:
            if not st["exists"]:
                missing.append(f"local model file {st['path']}")
    else:
        report["model_dir"]["warning"] = "HF repo id or non-local path; fixture run may download/load payloads via upstream code"

    report["missing"] = missing
    report["ready_for_lowstep_fixture"] = not missing

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}; ready={report['ready_for_lowstep_fixture']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
