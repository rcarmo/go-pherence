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
PODCAST_SHA = "8a7f5ea6b05a686ef1a6455d2a1683ed6cd1497a524efcb2cbfe10e810df6601"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, choices=(11, 100))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--podcast", action="store_true", help="real 20-second podcast crop at 300 seconds")
    parser.add_argument("--mixed", action="store_true", help="JFK followed by the podcast crop (31 seconds)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--segments-out", type=Path, help="processor segments on the same streaming logits")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.podcast and args.mixed:
        raise ValueError("select only one audio source")
    source = ROOT / "testdata/podcast.wav" if args.podcast else WAV
    if hashlib.sha256(source.read_bytes()).hexdigest() != (PODCAST_SHA if args.podcast else INPUT_SHA):
        raise ValueError("audio fixture provenance changed")
    audio, rate = sf.read(source, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or not args.podcast and len(audio) != 176000:
        raise ValueError("unexpected audio geometry")
    if args.podcast or args.mixed:
        if args.samples is not None or args.seconds is not None:
            raise ValueError("mixed/podcast audio selects its own duration")
        if args.mixed:
            source = ROOT / "testdata/podcast.wav"
            if hashlib.sha256(source.read_bytes()).hexdigest() != PODCAST_SHA:
                raise ValueError("podcast fixture provenance changed")
            podcast, podcast_rate = sf.read(source, dtype="float32", start=300*rate, stop=320*rate)
            if podcast_rate != rate or len(podcast) != 20*rate:
                raise ValueError("unexpected podcast crop")
            import numpy as np
            audio = np.concatenate((audio, podcast))
        else:
            audio = audio[300*rate:320*rate]
            if len(audio) != 20*rate:
                raise ValueError("unexpected podcast crop")
    else:
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
    joined = torch.cat(logits)
    output = joined.contiguous().numpy().astype("<f4", copy=False)
    if args.segments_out:
        args.segments_out.parent.mkdir(parents=True, exist_ok=True)
        segments = processor.extract_speaker_dict(joined[None, :, :])[0]
        args.segments_out.write_text(json.dumps(segments, indent=2) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(output.tobytes())
    print(json.dumps({"seconds": args.seconds, "samples": len(audio), "output": list(output.shape), "chunks": shapes}))


if __name__ == "__main__":
    main()
