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
from transformers import AutoModelForRNNT, AutoProcessor, DynamicCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import NemotronAsrStreamingEncoderCausalConvPaddingCache
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import chunked_limited_mask_function

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
                qkv_k = layer.self_attn.k_proj(attn_normal).view(1, 5, 8, 128).transpose(1, 2)
                qkv_v = layer.self_attn.v_proj(attn_normal).view(1, 5, 8, 128).transpose(1, 2)
                kv_cache = DynamicCache(config=model.encoder.config)
                for index, bounds in enumerate(((0, 1), (1, 3), (3, 5))):
                    keys, values = kv_cache.update(
                        qkv_k[:, :, bounds[0]:bounds[1]],
                        qkv_v[:, :, bounds[0]:bounds[1]],
                        layer_idx=0,
                    )
                    for kind, tensor in (("returned_k", keys), ("returned_v", values),
                                         ("state_k", kv_cache.layers[0].keys), ("state_v", kv_cache.layers[0].values)):
                        save(args.out / f"encoder0_attn_cache_{kind}{index}.f32.gz", tensor[0])
                # Run attention with a fresh cache per mode. The positions span
                # cached+current keys; query offsets remain global across chunks.
                for lookahead in (0, 3):
                    stream_cache = DynamicCache(config=model.encoder.config)
                    for index, bounds in enumerate(((0, 1), (1, 3), (3, 5))):
                        start, end = bounds
                        chunk = attn_normal[:, start:end]
                        chunk_positions = model.encoder.encode_positions(chunk, cached_frames=start)
                        save(args.out / f"encoder0_attn_chunk_positions_look{lookahead}_{index}.f32.gz", chunk_positions[0])
                        left_ctx, right_ctx = model.encoder._resolve_attn_context(lookahead)
                        chunk_mask = create_bidirectional_mask(
                            config=model.encoder.config,
                            inputs_embeds=chunk,
                            attention_mask=torch.ones((1, end), dtype=torch.bool),
                            past_key_values=stream_cache,
                            position_ids=torch.arange(start, end)[None, :],
                            and_mask_function=chunked_limited_mask_function(left_ctx, right_ctx),
                        )
                        if chunk_mask is None:
                            chunk_mask = torch.ones((1, 1, end-start, end), dtype=torch.bool)
                        save(args.out / f"encoder0_attn_chunk_mask_look{lookahead}_{index}.f32.gz", chunk_mask[0, 0].float())
                        chunk_output, _ = layer.self_attn(
                            chunk, position_embeddings=chunk_positions,
                            attention_mask=chunk_mask, past_key_values=stream_cache,
                        )
                        save(args.out / f"encoder0_attn_chunk_output_look{lookahead}_{index}.f32.gz", chunk_output[0])
                positions = model.encoder.encode_positions(ff1_residual)
                save(args.out / "encoder0_attn_positions.f32.gz", positions[0])
                attention, _ = layer.self_attn(attn_normal, position_embeddings=positions, attention_mask=None)
                save(args.out / "encoder0_attn_output.f32.gz", attention[0])
                attention_residual = ff1_residual + attention
                save(args.out / "encoder0_attn_residual.f32.gz", attention_residual[0])
                for lookahead in (0, 3):
                    left_ctx, right_ctx = model.encoder._resolve_attn_context(lookahead)
                    chunk_mask = create_bidirectional_mask(
                        config=model.encoder.config,
                        inputs_embeds=ff1_residual,
                        attention_mask=torch.ones((1, ff1_residual.shape[1]), dtype=torch.bool),
                        and_mask_function=chunked_limited_mask_function(left_ctx, right_ctx),
                    )
                    save(args.out / f"encoder0_attn_mask_look{lookahead}.f32.gz", chunk_mask[0, 0].float())
                    masked_attention, _ = layer.self_attn(attn_normal, position_embeddings=positions, attention_mask=chunk_mask)
                    save(args.out / f"encoder0_attn_output_look{lookahead}.f32.gz", masked_attention[0])
                    masked_residual = ff1_residual + masked_attention
                    save(args.out / f"encoder0_attn_residual_look{lookahead}.f32.gz", masked_residual[0])
                    masked_conv = layer.conv(layer.norm_conv(masked_residual))
                    masked_conv_residual = masked_residual + masked_conv
                    masked_ff2 = layer.feed_forward2(layer.norm_feed_forward2(masked_conv_residual))
                    save(args.out / f"encoder0_block_output_look{lookahead}.f32.gz", layer.norm_out(masked_conv_residual + 0.5 * masked_ff2)[0])
                conv_normal = layer.norm_conv(attention_residual)
                save(args.out / "encoder0_conv_normal.f32.gz", conv_normal[0])
                point1 = layer.conv.pointwise_conv1(conv_normal.transpose(1, 2))
                save(args.out / "encoder0_conv_point1.f32.gz", point1[0])
                glu = torch.nn.functional.glu(point1, dim=1)
                save(args.out / "encoder0_conv_glu.f32.gz", glu[0])
                cache = NemotronAsrStreamingEncoderCausalConvPaddingCache()
                for index, chunk in enumerate(glu.split((1, 2, 2), dim=2)):
                    padded = cache.update(chunk, layer.conv.depthwise_conv.cache_key, layer.conv.depthwise_conv)
                    save(args.out / f"encoder0_conv_cache_padded{index}.f32.gz", padded[0])
                    save(args.out / f"encoder0_conv_cache_state{index}.f32.gz", cache.layers[layer.conv.depthwise_conv.cache_key].cache[0])
                    chunk_depth = torch.nn.functional.conv1d(
                        padded,
                        layer.conv.depthwise_conv.weight,
                        layer.conv.depthwise_conv.bias,
                        groups=1024,
                    )
                    save(args.out / f"encoder0_conv_cache_depth{index}.f32.gz", chunk_depth[0])
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
