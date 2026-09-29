"""Generate long cached layer-0 ASR attention fixtures from released weights.

Prepared already-normalised rows isolate attention/KV sliding-window behaviour.
They do not qualify PCM scheduling or the complete 24-layer encoder.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import soundfile as sf

import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import chunked_limited_mask_function

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "checkpoints/nemotron/asr"
WAV = ROOT / "testdata/jfk.wav"
INPUT_SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def save(path, values):
    array = values.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    with path.open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(array.tobytes())
    return list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError("JFK fixture provenance changed")
    pcm, rate = sf.read(WAV, dtype="float32")
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(MODEL, local_files_only=True).eval()
    features = processor(pcm, sampling_rate=rate, return_tensors="pt").input_features
    layer = model.encoder.layers[0]
    with torch.inference_mode():
        embedded = model.encoder.subsampling(features[:, :128], torch.ones((1, 128), dtype=torch.long))
        # Repeat released-weight projected JFK rows as input to the cache probe.
        # Norm is applied with released layer weights, independently in PyTorch.
        repeats = (72 + embedded.shape[1] - 1) // embedded.shape[1]
        hidden = embedded.repeat(1, repeats, 1)[:, :72] * model.encoder.input_scale
        ff1 = hidden + .5*layer.feed_forward1(layer.norm_feed_forward1(hidden))
        normalised = layer.norm_self_att(ff1)
    if tuple(normalised.shape) != (1, 72, 1024):
        raise ValueError("unexpected layer-0 reference geometry")
    shapes = {"input": save(args.out / "input.f32.gz", normalised[0])}
    with torch.inference_mode():
        for lookahead in (0, 3):
            cache = DynamicCache(config=model.encoder.config)
            outputs = []
            bounds = []
            for start in range(0, 72, 4):
                end = start+4
                chunk = normalised[:, start:end]
                left, right = model.encoder._resolve_attn_context(lookahead)
                mask = create_bidirectional_mask(
                    config=model.encoder.config, inputs_embeds=chunk,
                    attention_mask=torch.ones((1, end), dtype=torch.bool),
                    past_key_values=cache, position_ids=torch.arange(start, end)[None, :],
                    and_mask_function=chunked_limited_mask_function(left, right),
                )
                pos = model.encoder.encode_positions(chunk, cached_frames=cache.get_seq_length())
                result, _ = layer.self_attn(chunk, position_embeddings=pos,
                                            attention_mask=mask, past_key_values=cache)
                outputs.append(result[0])
                bounds.append(int(cache.get_seq_length()))
            shapes[f"look{lookahead}"] = save(args.out / f"look{lookahead}.f32.gz", torch.cat(outputs))
            shapes[f"cache_seen_look{lookahead}"] = bounds
    print(json.dumps(shapes))


if __name__ == "__main__":
    main()
