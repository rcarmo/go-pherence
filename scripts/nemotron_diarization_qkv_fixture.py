"""Pin a 16-row first-layer offline window from the released diarization model.

Uses the local approved CPU Transformers environment and the independently
saved stacking fixture. Saves layer-0 and layer-1 attention stages;
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
    processed = processor(audio, sampling_rate=rate, return_tensors="pt")
    mask = processed.attention_mask
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
        full_layer0 = full_residual + layer.mlp(layer.layer_norm2(full_residual))
        save(args.out / "jfk_full_layer0_complete.f32.gz", full_layer0[0])
        layer1 = model.model.audio_tower.layers[1]
        full_layer1_normal = layer1.layer_norm1(full_layer0)
        save(args.out / "jfk_full_layer1_normal.f32.gz", full_layer1_normal[0])
        for name in ("q", "k", "v"):
            save(args.out / f"jfk_full_layer1_{name}.f32.gz", getattr(layer1.self_attn, f"{name}_proj")(full_layer1_normal)[0])
        full_layer1_attention, _ = layer1.self_attn(full_layer1_normal, position_embeddings=all_rotary, attention_mask=None)
        save(args.out / "jfk_full_layer1_attention.f32.gz", full_layer1_attention[0])
        full_layer1_residual = full_layer0 + full_layer1_attention
        save(args.out / "jfk_full_layer1_residual.f32.gz", full_layer1_residual[0])
        full_layer1_complete = full_layer1_residual + layer1.mlp(layer1.layer_norm2(full_layer1_residual))
        save(args.out / "jfk_full_layer1_complete.f32.gz", full_layer1_complete[0])
        layer2 = model.model.audio_tower.layers[2]
        full_layer2_normal = layer2.layer_norm1(full_layer1_complete)
        save(args.out / "jfk_full_layer2_normal.f32.gz", full_layer2_normal[0])
        full_layer2_attention, _ = layer2.self_attn(full_layer2_normal, position_embeddings=all_rotary, attention_mask=None)
        save(args.out / "jfk_full_layer2_attention.f32.gz", full_layer2_attention[0])
        full_layer2_residual = full_layer1_complete + full_layer2_attention
        save(args.out / "jfk_full_layer2_residual.f32.gz", full_layer2_residual[0])
        save(args.out / "jfk_full_layer2_complete.f32.gz", (full_layer2_residual + layer2.mlp(layer2.layer_norm2(full_layer2_residual)))[0])
        # The 31-layer offline tower is bidirectional over all 138 rows.
        # Save sparse intermediate checkpoints to detect accumulated drift.
        tower_hidden = all_normal
        for layer_index, tower_layer in enumerate(model.model.audio_tower.layers):
            tower_hidden = tower_layer(tower_hidden, position_embeddings=all_rotary, attention_mask=None)
            if layer_index in (7, 15, 23, 30):
                save(args.out / f"jfk_full_layer{layer_index}_complete.f32.gz", tower_hidden[0])
        tower_normal = model.model.audio_tower.layer_norm(tower_hidden)
        save(args.out / "jfk_full_tower_normal.f32.gz", tower_normal[0])
        projected = model.model.proj(tower_normal)
        save(args.out / "jfk_full_head_projected.f32.gz", projected[0])
        convolved = model.model.upsampler.conv(projected.transpose(1, 2))
        save(args.out / "jfk_full_head_convolved.f32.gz", convolved[0])
        upsampled = model.model.upsampler(projected)
        save(args.out / "jfk_full_head_upsampled.f32.gz", upsampled[0])
        logits = model.classifier(upsampled)
        save(args.out / "jfk_full_head_logits.f32.gz", logits[0])
        request = model(input_features=processed.input_features, attention_mask=mask)
        save(args.out / "jfk_request_logits.f32.gz", request.logits[0])
        # A separate 16-row window has its own bidirectional context.
        short_layer0 = (residual + layer.mlp(layer.layer_norm2(residual)))[None]
        short_layer1_normal = layer1.layer_norm1(short_layer0)
        short_layer1_attention, _ = layer1.self_attn(short_layer1_normal, position_embeddings=rotary, attention_mask=None)
        save(args.out / "jfk_layer1_attention.f32.gz", short_layer1_attention[0])
        short_layer1_residual = short_layer0 + short_layer1_attention
        save(args.out / "jfk_layer1_residual.f32.gz", short_layer1_residual[0])
        save(args.out / "jfk_layer1_complete.f32.gz", (short_layer1_residual + layer1.mlp(layer1.layer_norm2(short_layer1_residual)))[0])


if __name__ == "__main__":
    main()
