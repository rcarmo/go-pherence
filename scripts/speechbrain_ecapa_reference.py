#!/usr/bin/env python3
"""Generate SpeechBrain ECAPA reference embeddings for parity checks.

This optional development helper runs the upstream SpeechBrain EncoderClassifier
on an audio file and writes a JSON embedding. Use it to compare the Go
SpeechBrainECAPA forward/preprocessing path against the reference implementation.

Example:

    python3 -m pip install speechbrain torch torchaudio
    python3 scripts/speechbrain_ecapa_reference.py \
      --source speechbrain/spkrec-ecapa-voxceleb \
      --audio meeting.wav \
      --output /workspace/tmp/ecapa_reference.json

For local downloaded assets, pass the local directory as --source. The script is
not used by production Go code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fail_missing(exc: BaseException) -> None:
    print(
        "missing optional SpeechBrain reference dependency: %s\n"
        "install with: python3 -m pip install speechbrain torch torchaudio" % exc,
        file=sys.stderr,
    )
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="speechbrain/spkrec-ecapa-voxceleb", help="HF repo id or local SpeechBrain model directory")
    ap.add_argument("--audio", required=True, help="Audio file path")
    ap.add_argument("--output", default="", help="Optional JSON output path; stdout if omitted")
    ap.add_argument("--device", default="cpu", help="Torch device, default cpu")
    args = ap.parse_args()

    try:
        import torch
        import torchaudio
        import soundfile as sf
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as exc:  # pragma: no cover - optional helper
        fail_missing(exc)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"audio not found: {audio_path}", file=sys.stderr)
        return 1

    classifier = EncoderClassifier.from_hparams(
        source=args.source,
        savedir=str(Path("/workspace/tmp/speechbrain_ecapa_ref_cache")),
        run_opts={"device": args.device},
    )

    try:
        wav, sample_rate = torchaudio.load(str(audio_path))
    except Exception:
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T.copy())
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        sample_rate = 16000

    with torch.no_grad():
        emb = classifier.encode_batch(wav.to(args.device)).squeeze().detach().cpu().float()
    values = emb.tolist()
    payload = {
        "source": args.source,
        "audio": str(audio_path),
        "sample_rate": sample_rate,
        "dim": len(values),
        "embedding": values,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
