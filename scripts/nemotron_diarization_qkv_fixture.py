"""Pin a 16-row first-layer offline window from the released diarization model.

Uses the local approved CPU Transformers environment and the independently
saved stacking fixture. Saves pre-RoPE Q/K/V and full-window attention/residual;
no later encoder layers, upsampling or speaker head run here.
"""
import argparse
import gzip
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/diarization"
STACKED = ROOT / "model/nemotrondiarization/testdata/jfk_stacking_transformers_5_18.f32.gz"
WAV = ROOT / "testdata/jfk.wav"
INPUT_SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


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
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError("JFK input provenance changed")
    audio, rate = sf.read(WAV, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected audio geometry")
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    mask = processor(audio, sampling_rate=rate, return_tensors="pt").attention_mask
    if mask.shape != (1, 1101) or int(mask.sum()) != 1100 or not bool(mask[:, ::8].all()):
        raise ValueError("full attention window has an unexpected padding mask")
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
        residual = input_normal + attention[0]
        save(args.out / "jfk_layer0_residual.f32.gz", residual)
        save(args.out / "jfk_layer0_complete.f32.gz", residual + layer.mlp(layer.layer_norm2(residual)))
        all_rows = torch.from_numpy(stacked.copy().reshape(1, 138, 512))
        # The checked processor mask is valid at every downsampled position,
        # so this recording has no padding mask at the 138-row attention layer.
        all_normal = model.model.audio_tower.input_layer_norm(all_rows)
        all_positions = torch.arange(138)[None, :]
        all_rotary = model.model.audio_tower.rotary_emb(all_normal, all_positions)
        all_attention, _ = layer.self_attn(layer.layer_norm1(all_normal), position_embeddings=all_rotary, attention_mask=None)
        save(args.out / "jfk_full_layer0_attention.f32.gz", all_attention[0])
        full_residual = all_normal + all_attention
        save(args.out / "jfk_full_layer0_residual.f32.gz", full_residual[0])
        save(args.out / "jfk_full_layer0_complete.f32.gz", (full_residual + layer.mlp(layer.layer_norm2(full_residual)))[0])


if __name__ == "__main__":
    main()
