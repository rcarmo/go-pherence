"""Generate independent low-latency PCM-to-logits streaming fixtures.

Uses the pinned local CPU Transformers/PyTorch environment and released weights
from docs/validation/nemotron-speech-reference-2026-09-28.md. Outputs are not
bitwise acceptance gates: the Go test measures numerical error distributions.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, choices=(11, 100))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError("JFK fixture provenance changed")
    audio, rate = sf.read(WAV, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected audio geometry")
    if (args.seconds is None) == (args.samples is None):
        raise ValueError("specify one duration")
    if args.seconds == 100:
        import numpy as np
        audio = np.tile(audio, (args.seconds * rate + len(audio) - 1) // len(audio))[:args.seconds * rate]
    elif args.samples:
        audio = audio[:args.samples]
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL, local_files_only=True).eval()
    if processor.streaming_mode != "low_latency":
        raise ValueError("expected low_latency processor")
    cache = None
    logits = []
    shapes = []
    step = processor.num_mel_frames_per_step
    first_end = processor.num_samples_first_audio_chunk
    with torch.inference_mode():
        first_is_last = len(audio) < first_end
        first = processor(audio[:first_end], sampling_rate=rate, is_streaming=True,
                          is_first_audio_chunk=True, is_last_audio_chunk=first_is_last,
                          return_tensors="pt")
        result = model(**first, speaker_cache=cache)
        cache = result.speaker_cache
        logits.append(result.logits[0].cpu())
        shapes.append([int(first.input_features.shape[1]), int(result.logits.shape[1])])
        mel_frame_idx = step
        start = processor.audio_chunk_start(mel_frame_idx)
        while not first_is_last and start + processor.num_samples_per_audio_chunk <= len(audio):
            inp = processor(audio[start:start + processor.num_samples_per_audio_chunk],
                            sampling_rate=rate, is_streaming=True,
                            is_first_audio_chunk=False, return_tensors="pt")
            result = model(**inp, speaker_cache=cache)
            cache = result.speaker_cache
            logits.append(result.logits[0].cpu())
            shapes.append([int(inp.input_features.shape[1]), int(result.logits.shape[1])])
            mel_frame_idx += step
            start = processor.audio_chunk_start(mel_frame_idx)
        if not first_is_last:
            inp = processor(audio[start:], sampling_rate=rate, is_streaming=True,
                            is_first_audio_chunk=False, is_last_audio_chunk=True,
                            return_tensors="pt")
            result = model(**inp, speaker_cache=cache)
            logits.append(result.logits[0].cpu())
            shapes.append([int(inp.input_features.shape[1]), int(result.logits.shape[1])])
    output = torch.cat(logits).contiguous().numpy().astype("<f4", copy=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(output.tobytes())
    print(json.dumps({"seconds": args.seconds, "samples": len(audio), "output": list(output.shape), "chunks": shapes}))


if __name__ == "__main__":
    main()
