"""Export pinned diarization feature-stack inputs for bounded GPU operator tests.

Use the approved checkpoint and the pinned CPU Transformers environment.
Only the PyTorch-generated frontend tensor and released projection weights are
exported; model-output hashes are not used as acceptance gates.
"""
import argparse
import hashlib
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "testdata/jfk.wav"
MODEL = ROOT / "checkpoints/nemotron/diarization"
SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != SHA:
        raise ValueError("JFK input provenance changed")
    audio, rate = sf.read(WAV, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected input geometry")
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL, local_files_only=True).eval()
    features = processor(audio, sampling_rate=rate, return_tensors="pt").input_features[0]
    weight = model.model.audio_tower.embedder.projection.weight
    if tuple(features.shape) != (1101, 128) or tuple(weight.shape) != (512, 1024):
        raise ValueError("unexpected released stacking geometry")
    stacked = torch.nn.functional.pad(features, (0, 0, 0, 3)).reshape(138, 1024)
    (args.out / "stacked.f32").write_bytes(stacked.contiguous().numpy().astype("<f4", copy=False).tobytes())
    (args.out / "weight.f32").write_bytes(weight.detach().contiguous().numpy().astype("<f4", copy=False).tobytes())
    print("exported stacked [138,1024] and weight [512,1024]")


if __name__ == "__main__":
    main()
