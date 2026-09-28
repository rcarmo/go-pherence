"""Regenerate the pinned JFK Nemotron frontend fixture with an independent CPU reference.

Opt-in: install PyTorch CPU, librosa, soundfile and the pinned Transformers revision
listed in docs/validation/nemotron-speech-reference-2026-09-28.md. Never use
model-output byte equality as an acceptance criterion.
"""
import gzip
import hashlib
from pathlib import Path

import soundfile as sf
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "testdata/jfk.wav"
PROCESSOR = ROOT / "checkpoints/nemotron/asr"
OUTPUT = ROOT / "loader/audio/testdata/nemotron_jfk_transformers_5_18_features.f32.gz"
INPUT_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def main():
    if hashlib.sha256(INPUT.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("JFK fixture provenance changed")
    audio, sample_rate = sf.read(INPUT, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected fixture shape")
    processor = AutoProcessor.from_pretrained(PROCESSOR, local_files_only=True)
    features = processor.feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    tensor = features.input_features[0].contiguous().numpy().astype("<f4", copy=False)
    if tensor.shape != (1101, 128) or int(features.attention_mask[0].sum()) != 1100:
        raise ValueError("unexpected frontend shape or mask")
    with OUTPUT.open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
            compressed.write(tensor.tobytes())
    print(f"wrote {OUTPUT}: {tensor.shape}, valid frames=1100")


if __name__ == "__main__":
    main()
