"""Export bounded ASR stem patches and released weights for GPU operator probes.

Pinned CPU reference inputs come from scripts/nemotron_asr_stem_fixture.py.
The exported tensors stay outside Git; generated-output hashes are not gates.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "model/nemotronasr/testdata"


def load(name, shape):
    with gzip.open(FIXTURE / f"{name}.f32.gz", "rb") as source:
        raw = source.read()
    values = np.frombuffer(raw, dtype="<f4")
    if values.size != int(np.prod(shape)):
        raise ValueError(f"unexpected {name} fixture shape")
    return values.reshape(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    features = load("features", (32, 128))
    weight = load("weight", (256, 1, 3, 3)).reshape(256, 9)
    bias = load("bias", (256,))
    patches = np.zeros((17 * 65, 9), dtype="<f4")
    for row in range(17):
        for col in range(65):
            for kt in range(3):
                source_row = row * 2 + kt - 2
                if source_row < 0 or source_row >= 32:
                    continue
                for kf in range(3):
                    source_col = col * 2 + kf - 2
                    if 0 <= source_col < 128:
                        patches[row * 65 + col, kt * 3 + kf] = features[source_row, source_col]
    (args.out / "patches.f32").write_bytes(patches.tobytes())
    (args.out / "weight.f32").write_bytes(weight.astype("<f4", copy=False).tobytes())
    (args.out / "bias.f32").write_bytes(bias.astype("<f4", copy=False).tobytes())
    print("exported patches [1105,9], weight [256,9], bias [256]")


if __name__ == "__main__":
    main()
