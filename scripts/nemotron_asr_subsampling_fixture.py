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
            print("input valid", valid, "final valid length", int(lengths[0]))


if __name__ == "__main__":
    main()
