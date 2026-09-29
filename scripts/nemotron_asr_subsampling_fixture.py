"""Regenerate 32-frame ASR subsampling stage fixtures using pinned CPU Transformers.

Requires the approved local Nemotron 3.5 ASR checkpoint and the Transformers
revision in docs/validation/nemotron-speech-reference-2026-09-28.md.
"""
import argparse
import gzip
import hashlib
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "testdata/jfk.wav"
MODEL = ROOT / "checkpoints/nemotron/asr"
SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


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
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != SHA:
        raise ValueError("JFK fixture provenance changed")
    audio, rate = sf.read(WAV, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected audio")
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(MODEL, local_files_only=True).eval()
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
    features = inputs.input_features[:, :32, :]
    mask = inputs.attention_mask[:, :32]
    sub = model.encoder.subsampling
    with torch.inference_mode():
        for valid in (int(mask.sum(-1)[0]), 0, 1, 16, 31):
            lengths = torch.tensor([valid], dtype=torch.long)
            stem = sub.conv_in(features.unsqueeze(1))
            lengths = sub.conv_in.output_length(lengths)
            stem = sub.act_fn(stem * (torch.arange(stem.shape[2]) < lengths[:, None])[:, None, :, None])
            if valid == 32:
                save(args.out / "stem_activated.f32.gz", stem[0])
            hidden = stem
            for idx, layer in enumerate(sub.layers):
                hidden, lengths = layer(hidden, lengths)
                hidden = sub.act_fn(hidden)
                if valid == 32:
                    save(args.out / f"stage{idx}_activated.f32.gz", hidden[0])
            final = sub.linear(hidden.transpose(1, 2).reshape(1, hidden.shape[2], -1))
            name = "projected.f32.gz" if valid == 32 else f"projected_valid{valid}.f32.gz"
            save(args.out / name, final[0])
            if valid == 32:
                layer = model.encoder.layers[0]
                normal = layer.norm_feed_forward1(final)
                save(args.out / "encoder0_ff1_normal.f32.gz", normal[0])
                first = layer.feed_forward1.linear1(normal)
                save(args.out / "encoder0_ff1_linear1.f32.gz", first[0])
                activated = layer.feed_forward1.activation(first)
                save(args.out / "encoder0_ff1_activated.f32.gz", activated[0])
                feed_forward = layer.feed_forward1.linear2(activated)
                save(args.out / "encoder0_ff1_output.f32.gz", feed_forward[0])
                ff1_residual = final + 0.5 * feed_forward
                save(args.out / "encoder0_ff1_residual.f32.gz", ff1_residual[0])
                attn_normal = layer.norm_self_att(ff1_residual)
                save(args.out / "encoder0_attn_normal.f32.gz", attn_normal[0])
                for name in ("q", "k", "v"):
                    save(args.out / f"encoder0_attn_{name}.f32.gz", getattr(layer.self_attn, f"{name}_proj")(attn_normal)[0])
                positions = model.encoder.encode_positions(ff1_residual)
                save(args.out / "encoder0_attn_positions.f32.gz", positions[0])
                attention, _ = layer.self_attn(attn_normal, position_embeddings=positions, attention_mask=None)
                save(args.out / "encoder0_attn_output.f32.gz", attention[0])
                attention_residual = ff1_residual + attention
                save(args.out / "encoder0_attn_residual.f32.gz", attention_residual[0])
                conv_normal = layer.norm_conv(attention_residual)
                save(args.out / "encoder0_conv_normal.f32.gz", conv_normal[0])
                point1 = layer.conv.pointwise_conv1(conv_normal.transpose(1, 2))
                save(args.out / "encoder0_conv_point1.f32.gz", point1[0])
                glu = torch.nn.functional.glu(point1, dim=1)
                save(args.out / "encoder0_conv_glu.f32.gz", glu[0])
                depth = layer.conv.depthwise_conv(glu)
                save(args.out / "encoder0_conv_depth.f32.gz", depth[0])
                normalized_depth = layer.conv.norm(depth.transpose(1, 2))
                save(args.out / "encoder0_conv_depth_normal.f32.gz", normalized_depth[0])
                activated_depth = layer.conv.activation(normalized_depth).transpose(1, 2)
                save(args.out / "encoder0_conv_activated.f32.gz", activated_depth[0])
                conv_output = layer.conv.pointwise_conv2(activated_depth).transpose(1, 2)
                save(args.out / "encoder0_conv_output.f32.gz", conv_output[0])
                conv_residual = attention_residual + conv_output
                save(args.out / "encoder0_conv_residual.f32.gz", conv_residual[0])
                ff2_normal = layer.norm_feed_forward2(conv_residual)
                save(args.out / "encoder0_ff2_normal.f32.gz", ff2_normal[0])
                ff2_first = layer.feed_forward2.linear1(ff2_normal)
                save(args.out / "encoder0_ff2_linear1.f32.gz", ff2_first[0])
                ff2_activated = layer.feed_forward2.activation(ff2_first)
                save(args.out / "encoder0_ff2_activated.f32.gz", ff2_activated[0])
                ff2_output = layer.feed_forward2.linear2(ff2_activated)
                save(args.out / "encoder0_ff2_output.f32.gz", ff2_output[0])
                ff2_residual = conv_residual + 0.5 * ff2_output
                save(args.out / "encoder0_ff2_residual.f32.gz", ff2_residual[0])
                save(args.out / "encoder0_block_output.f32.gz", layer.norm_out(ff2_residual)[0])
            print("input valid", valid, "final valid length", int(lengths[0]))


if __name__ == "__main__":
    main()
