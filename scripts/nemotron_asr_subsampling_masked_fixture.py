"""Pinned PyTorch subsampling oracle for first and terminal masked JFK chunks.

The 13,000-sample prefix yields 25/26 first mel frames, 32 valid middle
frames and 24/32 valid terminal frames. This qualifies subsampling only.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np

import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import NemotronAsrStreamingEncoderCausalConvPaddingCache

ROOT = Path(__file__).resolve().parents[1]


def save(path, array):
    with path.open("wb") as file:
        with gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(array.contiguous().numpy().astype("<f4", copy=False).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-features", type=Path, help="rerun pinned subsampler on native frontend features")
    args = parser.parse_args()
    torch.set_num_threads(4)
    audio, rate = sf.read(ROOT / "testdata/jfk.wav", dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != 176000:
        raise ValueError("unexpected JFK geometry")
    processor = AutoProcessor.from_pretrained(ROOT / "checkpoints/nemotron/asr", local_files_only=True)
    model = AutoModelForRNNT.from_pretrained(ROOT / "checkpoints/nemotron/asr", local_files_only=True).eval()
    # Centered first chunk returns 26 frames with 25 valid. Subsequent
    # center=False windows overlap by n_fft/2, making frame 25 valid again.
    # The terminal window needs 56 zero PCM samples to complete frame 80.
    first = processor(audio[:4040], sampling_rate=rate, is_streaming=True,
                      is_first_audio_chunk=True, return_tensors="pt")
    middle = processor(audio[3744:9264], sampling_rate=rate, is_streaming=True,
                       is_first_audio_chunk=False, return_tensors="pt")
    terminal_audio = torch.cat([torch.from_numpy(audio[8864:13000]), torch.zeros(56)]).numpy()
    last = processor(terminal_audio, sampling_rate=rate, is_streaming=True,
                     is_first_audio_chunk=False, return_tensors="pt")
    if [list(x.input_features.shape) for x in (first, middle, last)] != [[1, 26, 128], [1, 32, 128], [1, 24, 128]]:
        raise ValueError("unexpected processor chunk shape")
    if [int(first.attention_mask.sum()), int(middle.attention_mask.sum())] != [25, 32]:
        raise ValueError("unexpected processor mask")
    # Padding on the terminal audio is STFT support, not an extra mel frame.
    # Right-pad features to the regular 32-frame conv chunk, mask eight rows.
    feature_chunks = (first.input_features, middle.input_features,
                      torch.cat([last.input_features, torch.zeros(1, 8, 128)], dim=1))
    features = torch.cat(feature_chunks, dim=1)
    masks = [25, 32, 24]
    offline = processor(audio[:13000], sampling_rate=rate, return_tensors="pt").input_features
    # Validate that the masked first row and the overlapped next chunk are
    # exactly at the expected offline frame boundaries, up to STFT rounding.
    for chunk, start, valid in zip(feature_chunks, (0, 25, 57), masks):
        if not torch.allclose(chunk[:, :valid], offline[:, start:start + valid], atol=3e-4, rtol=2e-5):
            raise ValueError(f"streaming processor differs at mel row {start}")
    cache = NemotronAsrStreamingEncoderCausalConvPaddingCache()
    outputs = []
    with torch.inference_mode():
        for chunk, valid in zip(feature_chunks, masks):
            mask = torch.arange(chunk.shape[1])[None, :] < valid
            outputs.append(model.encoder.subsampling(chunk, mask, cache)[0].cpu())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, torch.cat(outputs))
    feature_path = args.out.with_name(args.out.name.replace(".f32.gz", ".features.f32.gz"))
    save(feature_path, features[0])
    if args.native_features:
        with gzip.open(args.native_features, "rb") as file:
            native = np.frombuffer(file.read(), dtype="<f4").copy()
        if native.size != 90 * 128:
            raise ValueError("native feature shape")
        native_cache = NemotronAsrStreamingEncoderCausalConvPaddingCache()
        native_outputs = []
        with torch.inference_mode():
            offset = 0
            for rows, valid in zip((26, 32, 32), masks):
                chunk = torch.from_numpy(native[offset * 128:(offset + rows) * 128].reshape(1, rows, 128))
                mask = torch.arange(rows)[None, :] < valid
                native_outputs.append(model.encoder.subsampling(chunk, mask, native_cache)[0].cpu())
                offset += rows
        native_path = args.out.with_name(args.out.name.replace(".f32.gz", ".native-input-ref.f32.gz"))
        save(native_path, torch.cat(native_outputs))
    print(json.dumps({"input": list(offline.shape), "valid": masks,
                      "chunks": [list(c.shape) for c in feature_chunks],
                      "output": [list(o.shape) for o in outputs], "features": str(feature_path)}))


if __name__ == "__main__":
    main()
