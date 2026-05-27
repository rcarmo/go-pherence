#!/usr/bin/env python3
"""Run a Hugging Face Transformers Whisper reference decode for parity checks.

This script is intentionally optional: it is not part of the Go build and only
runs when the Python reference stack is installed, e.g.:

    python3 -m pip install 'transformers>=4.40' torch librosa soundfile accelerate

Example:

    python3 scripts/whisper_transformers_reference.py \
      --model models/whisper-large-v3-turbo-hf \
      --audio /path/to/clip.wav \
      --language es \
      --task translate \
      --chunk-offset 0 \
      --chunk-seconds 10

Use this to compare Go Whisper prompt/task behavior against the upstream
Transformers generation path before enabling turbo/distilled checkpoints as
production defaults.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def fail_missing(exc: BaseException) -> None:
    print(
        "missing optional Python reference dependency: %s\n"
        "install with: python3 -m pip install 'transformers>=4.40' torch librosa soundfile accelerate"
        % exc,
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model id or local model directory")
    ap.add_argument("--audio", required=True, help="Audio file readable by librosa/soundfile")
    ap.add_argument("--language", default=None, help="Whisper language, e.g. es, pt, english")
    ap.add_argument("--task", default="translate", choices=("translate", "transcribe"))
    ap.add_argument("--chunk-offset", type=float, default=0.0, help="Seconds to skip before decoding")
    ap.add_argument("--chunk-seconds", type=float, default=0.0, help="Optional duration to decode")
    ap.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu")
    args = ap.parse_args()

    try:
        import torch
        import librosa
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except Exception as exc:  # pragma: no cover - optional diagnostic script
        fail_missing(exc)

    model_path = Path(args.model)
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"audio not found: {audio_path}", file=sys.stderr)
        return 1

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(str(model_path))

    samples, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
    if args.chunk_offset > 0 or args.chunk_seconds > 0:
        start = max(0, int(args.chunk_offset * sample_rate))
        end = len(samples)
        if args.chunk_seconds > 0:
            end = min(end, start + int(args.chunk_seconds * sample_rate))
        samples = samples[start:end]

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    generate_kwargs = {"task": args.task}
    if args.language:
        generate_kwargs["language"] = args.language

    result = pipe({"array": samples, "sampling_rate": sample_rate}, generate_kwargs=generate_kwargs)
    print(result["text"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
