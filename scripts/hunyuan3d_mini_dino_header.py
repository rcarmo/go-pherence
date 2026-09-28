#!/usr/bin/env python3
"""Freeze the pinned Hunyuan3D-2mini DINO metadata fixture (no tensor payloads).

Run from the repository root:
  PYTHONDONTWRITEBYTECODE=1 uv run --no-project python scripts/hunyuan3d_mini_dino_header.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from hunyuan3d_fixture_inventory import fetch_bytes, fetch_safetensors_header, resolve_url

REPO = "tencent/Hunyuan3D-2mini"
REVISION = "f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6"
SUBFOLDER = "hunyuan3d-dit-v2-mini"
CONFIG_SHA256 = "cabcba7f6115752c8fe5b370e12bf714936f70377a8a80f151872f76c2d64609"
HEADER_SHA256 = "dd8b61f43325eb7f58717bcfe0e4fafc3932cfef3bf4dee96272d52a757427b4"
PREFIX = "conditioner.main_image_encoder.model."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="model/hunyuan3d/testdata/mini_dino_header.json")
    args = parser.parse_args()
    config = fetch_bytes(resolve_url(REPO, f"{SUBFOLDER}/config.yaml", REVISION))
    if hashlib.sha256(config).hexdigest() != CONFIG_SHA256:
        raise RuntimeError("mini config digest changed")
    header = fetch_safetensors_header(REPO, f"{SUBFOLDER}/model.fp16.safetensors", REVISION)
    header_digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()
    if header_digest != HEADER_SHA256:
        raise RuntimeError("mini safetensors header digest changed")
    tensors = {
        name[len(PREFIX):]: {"dtype": value["dtype"], "shape": value["shape"]}
        for name, value in sorted(header.items())
        if name.startswith(PREFIX)
        and (".layer.0." in name or not name.startswith(PREFIX + "encoder.layer."))
    }
    all_conditioner = {
        name: {"dtype": value["dtype"], "shape": value["shape"]}
        for name, value in header.items() if name.startswith("conditioner.")
    }
    globals_ = {name: spec for name, spec in tensors.items() if not name.startswith("encoder.layer.")}
    layer_zero = {
        name.removeprefix("encoder.layer.0."): spec
        for name, spec in tensors.items() if name.startswith("encoder.layer.0.")
    }
    expected = {PREFIX + name: spec for name, spec in globals_.items()}
    for layer in range(40):
        expected.update({
            PREFIX + f"encoder.layer.{layer}.{name}": spec
            for name, spec in layer_zero.items()
        })
    if len(globals_) != 7 or len(layer_zero) != 18 or all_conditioner != expected:
        raise RuntimeError("unexpected mini conditioner tensor names, dtypes or shapes")
    fixture = {
        "schema": 1,
        "repository": REPO,
        "revision": REVISION,
        "subfolder": SUBFOLDER,
        "config_sha256": CONFIG_SHA256,
        "safetensors_header_sha256": HEADER_SHA256,
        "conditioner_count": len(all_conditioner),
        "layers": 40,
        "tensors": tensors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(tensors)} template tensors, {len(all_conditioner)} conditioner tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
