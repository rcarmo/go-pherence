"""Generate pinned PyTorch full-valid streaming subsampling reference on JFK PCM.

This tests Conv2D padding-cache transitions only; it does not qualify the
24-layer encoder, RNN-T transcription or a terminal masked chunk.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import NemotronAsrStreamingEncoderCausalConvPaddingCache

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "testdata/jfk.wav"
MODEL = ROOT / "checkpoints/nemotron/asr"
INPUT_SHA = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, choices=(8, 16, 32, 64), default=32)
    args = parser.parse_args()
    if hashlib.sha256(WAV.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError("JFK fixture provenance changed")
    audio, rate = sf.read(WAV, dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected JFK geometry")
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(MODEL, local_files_only=True).eval()
    features = processor(audio, sampling_rate=rate, return_tensors="pt").input_features
    if features.shape[1] < 128:
        raise ValueError("insufficient feature frames")
    cache = NemotronAsrStreamingEncoderCausalConvPaddingCache()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, 128, args.chunk):
            chunk = features[:, offset:offset+args.chunk]
            projected = model.encoder.subsampling(chunk, torch.ones(1, args.chunk, dtype=torch.long), cache)
            outputs.append(projected[0].cpu())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    values = torch.cat(outputs).contiguous().numpy().astype("<f4", copy=False)
    input_values = features[0, :128].contiguous().numpy().astype("<f4", copy=False)
    input_path = args.out.with_name(args.out.name.replace(".f32.gz", ".features.f32.gz"))
    for path, array in ((args.out, values), (input_path, input_values)):
        with path.open("wb") as file:
            with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
                compressed.write(array.tobytes())
    print(json.dumps({"feature_rows": 128, "chunk_size": args.chunk, "chunks": [list(o.shape) for o in outputs], "output": list(values.shape), "features": str(input_path)}))


if __name__ == "__main__":
    main()
