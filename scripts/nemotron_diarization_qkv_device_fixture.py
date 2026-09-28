"""Export pinned first-layer pre-RoPE Q/K/V GEMM inputs for GPU probes.

The normalized input fixture is an independent CPU Transformers output. The
combined weight stacks released Q, K, V rows; no normalization or attention
operation runs on the GPU in these probes.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/diarization/model.safetensors"
FIXTURE = ROOT / "model/nemotrondiarization/testdata/jfk_layer0_normal.f32.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(FIXTURE, "rb") as f:
        normal = np.frombuffer(f.read(), dtype="<f4")
    if normal.shape != (16 * 512,):
        raise ValueError("unexpected normalized input")
    normal.tofile(args.out / "normal.f32")
    with safe_open(MODEL, framework="pt", device="cpu") as f:
        rows = []
        for name in ("q", "k", "v"):
            rows.append(f.get_tensor(f"model.audio_tower.layers.0.self_attn.{name}_proj.weight").numpy())
    weights = np.concatenate(rows, axis=0).astype("<f4", copy=False)
    if weights.shape != (1536, 512) or not np.isfinite(weights).all():
        raise ValueError("unexpected QKV weight")
    weights.tofile(args.out / "weight.f32")
    print("normal", normal.shape, "weight", weights.shape)


if __name__ == "__main__":
    main()
