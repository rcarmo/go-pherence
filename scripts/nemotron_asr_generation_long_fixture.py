"""Pinned streaming-generate oracle on JFK or 100-second tiled JFK, lookahead three.

Use the full-recording processor for mel values, then the generation API's
25/32 schedule with zero right padding. This saves raw decisions and decoded
text, not a WER fixture. The WAV digest checks input provenance only.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "testdata/jfk.wav"
INPUT_SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, choices=(11, 100), default=11)
    args = parser.parse_args()
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError("JFK fixture provenance changed")
    pcm, rate = sf.read(WAV, dtype="float32")
    if rate != 16000 or pcm.ndim != 1 or len(pcm) != 176000:
        raise ValueError("unexpected JFK geometry")
    if args.seconds == 100:
        import numpy as np
        pcm = np.tile(pcm, (args.seconds * rate + len(pcm) - 1) // len(pcm))[:args.seconds * rate]
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(ROOT / "checkpoints/nemotron/asr", local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(ROOT / "checkpoints/nemotron/asr", local_files_only=True).eval()
    features = processor(pcm, sampling_rate=rate, return_tensors="pt")
    valid = args.seconds * 100
    if tuple(features.input_features.shape) != (1, valid + 1, 128) or int(features.attention_mask.sum()) != valid:
        raise ValueError("unexpected feature geometry")
    chunks = [features.input_features[:, :25]]
    for start in range(25, valid, 32):
        chunk = features.input_features[:, start:min(start + 32, valid)]
        if chunk.shape[1] < 32:
            chunk = torch.cat([chunk, torch.zeros(1, 32 - chunk.shape[1], 128)], dim=1)
        chunks.append(chunk)
    stages = {"input": [], "layer0": [], "tower": [], "normal0": [], "attention0": []}
    hooks = [
        model.encoder.subsampling.register_forward_hook(
            lambda module, inputs, output: stages["input"].append(output[0].detach().cpu())),
        model.encoder.layers[0].register_forward_hook(
            lambda module, inputs, output: stages["layer0"].append(output[0].detach().cpu())),
        model.encoder.layers[0].norm_self_att.register_forward_hook(
            lambda module, inputs, output: stages["normal0"].append(output[0].detach().cpu())),
        model.encoder.layers[0].self_attn.register_forward_hook(
            lambda module, inputs, output: stages["attention0"].append(output[0][0].detach().cpu())),
        model.encoder.register_forward_hook(
            lambda module, inputs, output: stages["tower"].append(output.last_hidden_state[0].detach().cpu())),
    ]
    with torch.inference_mode():
        output = model.generate(input_features=(chunk for chunk in chunks), num_lookahead_tokens=3,
                                max_new_tokens=16000)
    for hook in hooks:
        hook.remove()
    for name, tensors in stages.items():
        array = torch.cat(tensors).contiguous().numpy().astype("<f4", copy=False)
        with (args.out.parent / f"{args.out.stem}.{name}.f32.gz").open("wb") as file:
            with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
                compressed.write(array.tobytes())
    tokens = output.sequences[0].tolist()
    durations = output.durations[0].tolist()
    if len(tokens) != len(durations) or tokens[0] != model.config.blank_token_id:
        raise ValueError("unexpected generation decisions")
    text = processor.decode(tokens, skip_special_tokens=True)
    result = {"sample_rate": rate, "samples": len(pcm), "mel_valid": valid,
              "chunk_rows": [int(c.shape[1]) for c in chunks], "tokens": tokens,
              "durations": durations, "text": text, "nonblank": sum(t != model.config.blank_token_id for t in tokens)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"chunks": len(chunks), "decisions": len(tokens), "nonblank": result["nonblank"],
                      "text": text, "output": str(args.out)}))


if __name__ == "__main__":
    main()
