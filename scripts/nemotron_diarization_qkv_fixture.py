"""Pin 16 first-layer pre-attention rows from the released diarization model.

Uses the local approved CPU Transformers environment and the independently
saved stacking fixture. This is pre-RoPE Q/K/V only, not attention or inference.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/diarization"
STACKED = ROOT / "model/nemotrondiarization/testdata/jfk_stacking_transformers_5_18.f32.gz"


def save(path, tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    with path.open("wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as compressed:
            compressed.write(array.tobytes())
    print(path.name, array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(STACKED, "rb") as f:
        stacked = np.frombuffer(f.read(), dtype="<f4")
    if stacked.size != 138 * 512:
        raise ValueError("unexpected stacking fixture geometry")
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL, local_files_only=True).eval()
    layer = model.model.audio_tower.layers[0]
    with torch.inference_mode():
        inputs = torch.from_numpy(stacked[:16 * 512].copy().reshape(16, 512))
        input_normal = model.model.audio_tower.input_layer_norm(inputs)
        save(args.out / "jfk_input_normal.f32.gz", input_normal)
        normal = layer.layer_norm1(input_normal)
        save(args.out / "jfk_layer0_normal.f32.gz", normal)
        for name in ("q", "k", "v"):
            save(args.out / f"jfk_layer0_{name}.f32.gz", getattr(layer.self_attn, f"{name}_proj")(normal))
        positions = torch.arange(16)[None, :]
        rotary = model.model.audio_tower.rotary_emb(input_normal[None], positions)
        attention, _ = layer.self_attn(normal[None], position_embeddings=rotary, attention_mask=None)
        save(args.out / "jfk_layer0_attention.f32.gz", attention[0])
        save(args.out / "jfk_layer0_residual.f32.gz", input_normal + attention[0])


if __name__ == "__main__":
    main()
