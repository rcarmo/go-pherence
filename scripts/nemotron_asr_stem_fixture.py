"""Regenerate bounded ASR first-stem Conv2D fixtures from pinned CPU Transformers.

Requires the approved Nemotron ASR checkpoint and pinned Transformers revision
listed in docs/validation/nemotron-speech-reference-2026-09-28.md.
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
    return tuple(array.shape)


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
    # 32 frames include a valid+masked tail, while remaining small enough for
    # reference matmul/Conv2D output fixtures and CPU scalar regression checks.
    features = processor(audio, sampling_rate=rate, return_tensors="pt").input_features[:, :32, :]
    stem = model.encoder.subsampling.conv_in
    with torch.inference_mode():
        output = stem(features.unsqueeze(1), padding_cache=None)
    print("input", save(args.out / "features.f32.gz", features[0]))
    print("weight", save(args.out / "weight.f32.gz", stem.weight))
    print("bias", save(args.out / "bias.f32.gz", stem.bias))
    print("output", save(args.out / "output.f32.gz", output[0]))


if __name__ == "__main__":
    main()
