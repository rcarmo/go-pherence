#!/usr/bin/env python3
"""Generate a Hunyuan3D ImageProcessorV2 preprocessing fixture.

The script mirrors the upstream preprocessing path used before DINO/CLIP
conditioning:

1. ensure RGBA/mask input;
2. recenter the alpha/object mask with a configurable border;
3. composite RGB over white;
4. resize image with bicubic and mask with nearest-neighbour;
5. convert to BCHW float32 in [-1, 1].

It is dependency-gated because the Go repository remains pure/static by
default. Install local fixture dependencies with:

    python3 -m pip install numpy pillow opencv-python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def require_deps():
    missing = []
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
        missing.append("opencv-python")
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")
    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None
        missing.append("pillow")
    if missing:
        print(
            "missing Python fixture dependencies: "
            + ", ".join(missing)
            + "\ninstall with: python3 -m pip install numpy pillow opencv-python",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return cv2, np, Image


def synthetic_rgba(np, width: int, height: int):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    # Deliberately off-centre object so recentering is exercised.
    x0, x1 = width // 5, width // 5 + width // 3
    y0, y1 = height // 4, height // 4 + height // 2
    for y in range(y0, y1):
        for x in range(x0, x1):
            img[y, x, 0] = (37 + 3 * x + y) % 256
            img[y, x, 1] = (91 + x + 5 * y) % 256
            img[y, x, 2] = (173 + 2 * x + 7 * y) % 256
            img[y, x, 3] = 255
    return img


def recenter(cv2, np, image, border_ratio: float):
    if image.shape[-1] == 4:
        mask = image[..., 3]
    else:
        mask4 = np.ones_like(image[..., 0:1]) * 255
        image = np.concatenate([image, mask4], axis=-1)
        mask = mask4[..., 0]

    h, w, c = image.shape
    size = max(h, w)
    result = np.zeros((size, size, c), dtype=np.uint8)

    coords = np.nonzero(mask)
    if len(coords[0]) == 0 or len(coords[1]) == 0:
        raise ValueError("input image is empty")
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    obj_h = x_max - x_min
    obj_w = y_max - y_min
    if obj_h == 0 or obj_w == 0:
        raise ValueError("input image is empty")
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(obj_h, obj_w)
    h2 = int(obj_h * scale)
    w2 = int(obj_w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2

    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
        image[x_min:x_max, y_min:y_max],
        (w2, h2),
        interpolation=cv2.INTER_AREA,
    )

    bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255
    alpha = result[..., 3:].astype(np.float32) / 255
    rgb = result[..., :3] * alpha + bg * (1 - alpha)
    alpha = alpha * 255
    return rgb.clip(0, 255).astype(np.uint8), alpha.clip(0, 255).astype(np.uint8)


def array_to_bchw_tensor(np, arr):
    x = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    if x.ndim == 2:
        x = x[..., None]
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x.astype(np.float32)


def preprocess(cv2, np, image, size: int, border_ratio: float):
    image, mask = recenter(cv2, np, image, border_ratio)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 2:
        mask = mask[..., None]
    return array_to_bchw_tensor(np, image), array_to_bchw_tensor(np, mask), image, mask


def tensor_summary(np, name: str, value) -> dict[str, Any]:
    raw = value.astype("<f4", copy=False).tobytes()
    flat = value.reshape(-1)
    sample_count = min(16, flat.shape[0])
    return {
        "name": name,
        "dtype": "float32",
        "shape": list(value.shape),
        "sha256_le_f32": hashlib.sha256(raw).hexdigest(),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
        "mean": float(np.mean(value)),
        "first_values": [float(x) for x in flat[:sample_count]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="input image path; if omitted, a synthetic RGBA image is used")
    parser.add_argument("--out", default="testdata/hunyuan3d/image-preprocess-fixture.json")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--border-ratio", type=float, default=0.15)
    parser.add_argument("--synthetic-width", type=int, default=96)
    parser.add_argument("--synthetic-height", type=int, default=72)
    args = parser.parse_args()

    cv2, np, Image = require_deps()
    if args.image:
        pil = Image.open(args.image).convert("RGBA")
        source = np.asarray(pil)
        source_kind = "file"
        source_path = args.image
    else:
        source = synthetic_rgba(np, args.synthetic_width, args.synthetic_height)
        source_kind = "synthetic_rgba"
        source_path = ""

    image_tensor, mask_tensor, recentered_rgb, recentered_mask = preprocess(cv2, np, source, args.size, args.border_ratio)
    fixture = {
        "schema": "go-pherence-hunyuan3d-image-preprocess-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processor": "ImageProcessorV2",
        "source": {
            "kind": source_kind,
            "path": source_path,
            "shape": list(source.shape),
            "sha256_u8": hashlib.sha256(source.tobytes()).hexdigest(),
        },
        "params": {
            "size": args.size,
            "border_ratio": args.border_ratio,
            "image_resize": "cv2.INTER_CUBIC",
            "mask_resize": "cv2.INTER_NEAREST",
            "object_resize": "cv2.INTER_AREA",
            "tensor_layout": "BCHW",
            "tensor_range": [-1.0, 1.0],
        },
        "outputs": [
            tensor_summary(np, "image", image_tensor),
            tensor_summary(np, "mask", mask_tensor),
        ],
        "debug_u8": {
            "recentered_rgb_shape": list(recentered_rgb.shape),
            "recentered_mask_shape": list(recentered_mask.shape),
            "recentered_rgb_sha256": hashlib.sha256(recentered_rgb.tobytes()).hexdigest(),
            "recentered_mask_sha256": hashlib.sha256(recentered_mask.tobytes()).hexdigest(),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
