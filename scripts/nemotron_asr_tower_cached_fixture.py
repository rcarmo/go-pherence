"""Generate released-weight 24-layer cached encoder references on prepared rows.

Four JFK-derived projected rows are prepared with pinned subsampling and
replayed to exercise cached four-row steps. This is not PCM transcription.
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
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderCausalConvPaddingCache, chunked_limited_mask_function,
)

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "testdata/jfk.wav"
MODEL = ROOT / "checkpoints/nemotron/asr"
INPUT_SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def save(path, tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    with path.open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(array.tobytes())
    return list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=(8,72), default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError("JFK fixture provenance changed")
    pcm, rate = sf.read(WAV, dtype="float32")
    if rate != 16000 or len(pcm) != 176000:
        raise ValueError("unexpected JFK geometry")
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(MODEL, local_files_only=True).eval()
    features = processor(pcm, sampling_rate=rate, return_tensors="pt").input_features
    with torch.inference_mode():
        source = model.encoder.subsampling(features[:, :32], torch.ones((1,32),dtype=torch.long))
        rows = source[:, :4].repeat(1, args.rows//4, 1) * model.encoder.input_scale
        if tuple(rows.shape) != (1, args.rows, 1024):
            raise ValueError("unexpected prepared rows")
        shapes = {"input": save(args.out / "input.f32.gz", rows[0])}
        for lookahead in (0, 3):
            kv = DynamicCache(config=model.encoder.config)
            conv = NemotronAsrStreamingEncoderCausalConvPaddingCache()
            outputs = []
            for step in range(args.rows//4):
                start = step*4
                chunk = rows[:, start:start+4]
                left, right = model.encoder._resolve_attn_context(lookahead)
                mask = create_bidirectional_mask(
                    config=model.encoder.config, inputs_embeds=chunk,
                    attention_mask=torch.ones((1, start+4), dtype=torch.bool),
                    past_key_values=kv, position_ids=torch.arange(start,start+4)[None,:],
                    and_mask_function=chunked_limited_mask_function(left,right),
                )
                pos = model.encoder.encode_positions(chunk, cached_frames=kv.get_seq_length())
                hidden = chunk
                for layer in model.encoder.layers:
                    hidden = layer(hidden, attention_mask=mask, position_embeddings=pos,
                                   past_key_values=kv, padding_cache=conv)
                outputs.append(hidden[0])
                if kv.get_seq_length() != start+4:
                    raise ValueError("cache did not advance")
            shapes[f"look{lookahead}"] = save(args.out / f"look{lookahead}.f32.gz", torch.cat(outputs))
    print(json.dumps(shapes))


if __name__ == "__main__":
    main()
