"""Pin independent 100-second Nemotron mel features from tiled JFK PCM.

This intentionally tests a full-duration Transformers frontend against a Go
streaming frontend. It does not run either model or assert WER/DER.
"""
import argparse
import gzip
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForAudioFrameClassification

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "testdata/jfk.wav"
MODEL = ROOT / "checkpoints/nemotron/asr"
INPUT_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "loader/audio/testdata/nemotron_jfk_loop100_transformers_5_18_features.f32.gz")
    parser.add_argument("--diar-stacked-out", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(INPUT.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("JFK input provenance changed")
    pcm, rate = sf.read(INPUT, dtype="float32")
    if rate != 16000 or pcm.shape != (176000,):
        raise ValueError("unexpected JFK audio")
    tiled = np.tile(pcm, 10)[:1600000]
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    features = processor.feature_extractor(tiled, sampling_rate=rate, return_tensors="pt")
    result = features.input_features[0].contiguous().numpy().astype("<f4", copy=False)
    if result.shape != (10001, 128) or int(features.attention_mask[0].sum()) != 10000:
        raise ValueError("unexpected long frontend shape or mask")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as out:
        with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as gz:
            gz.write(result.tobytes())
    print(args.out, result.shape)
    if args.diar_stacked_out is not None:
        diar_model = AutoModelForAudioFrameClassification.from_pretrained(
            ROOT / "checkpoints/nemotron/diarization", local_files_only=True).eval()
        with torch.inference_mode():
            projected = diar_model.model.audio_tower.embedder(features.input_features)[0]
        if projected.shape != (1251, 512):
            raise ValueError(f"unexpected stacked shape {projected.shape}")
        array = projected.contiguous().numpy().astype("<f4", copy=False)
        args.diar_stacked_out.parent.mkdir(parents=True, exist_ok=True)
        with args.diar_stacked_out.open("wb") as out:
            with gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as gz:
                gz.write(array.tobytes())
        print(args.diar_stacked_out, array.shape)


if __name__ == "__main__":
    main()
