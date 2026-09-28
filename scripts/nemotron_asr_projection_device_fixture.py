"""Export released ASR 5×4352 final-projection inputs for GPU operator probes.

The channel-major stage-1 fixture is an independent CPU Transformers output.
This exporter does not implement the preceding causal convolutions on GPU.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/asr/model.safetensors"
STAGE = ROOT / "model/nemotronasr/testdata/stage1_activated.f32.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(STAGE, "rb") as file:
        hidden = np.frombuffer(file.read(), dtype="<f4")
    if hidden.shape != (256 * 5 * 17,):
        raise ValueError("unexpected stage-1 geometry")
    flat = np.ascontiguousarray(hidden.reshape(256, 5, 17).transpose(1, 0, 2).reshape(5, 4352))
    flat.tofile(args.out / "input.f32")
    with safe_open(MODEL, framework="pt", device="cpu") as file:
        weight = file.get_tensor("encoder.subsampling.linear.weight").numpy()
        bias = file.get_tensor("encoder.subsampling.linear.bias").numpy()
    if weight.shape != (1024, 4352) or bias.shape != (1024,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("unexpected final projection geometry or weight")
    weight.astype("<f4", copy=False).tofile(args.out / "weight.f32")
    bias.astype("<f4", copy=False).tofile(args.out / "bias.f32")
    print("input", flat.shape, "weight", weight.shape, "bias", bias.shape)


if __name__ == "__main__":
    main()
