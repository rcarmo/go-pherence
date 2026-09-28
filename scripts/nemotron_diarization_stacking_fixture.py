"""Regenerate a bounded independent Nemotron diarization feature-stack fixture.

Requires the pinned CPU Transformers/PyTorch environment and approved local
checkpoint in docs/validation/nemotron-speech-reference-2026-09-28.md.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "testdata/jfk.wav"
MODEL = ROOT / "checkpoints/nemotron/diarization"
INPUT_SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def save(path, tensor):
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    with path.open("wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as compressed:
            compressed.write(array.tobytes())
    return list(array.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError("JFK fixture provenance changed")
    audio, rate = sf.read(WAV, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected input geometry")
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL, local_files_only=True).eval()
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
    with torch.inference_mode():
        projected = model.model.audio_tower.embedder(inputs.input_features)
    shapes = {
        "input": save(args.out / "features.f32.gz", inputs.input_features[0]),
        "output": save(args.out / "projected.f32.gz", projected[0]),
    }
    print(json.dumps(shapes))


if __name__ == "__main__":
    main()
