#!/usr/bin/env python3
"""Export deterministic MOSS Whisper boundary samples from the pinned checkpoint.

This is a fixture generator only; native inference never invokes Python.
Requires numpy, torch, transformers==4.57.1, and safetensors.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

PREFIX = "model.whisper_encoder."
POSITIONS = (0, 1, 17, 749, 1499)
DIMS = (0, 1, 17, 511, 1023)
CHANNELS = (0, 1, 17, 511, 1023)
FRAMES = (0, 1, 17, 1499, 2999)


def deterministic_waveform() -> np.ndarray:
    samples = np.zeros(30 * 16000, dtype=np.float32)
    for i in range(16000):
        samples[i] = 0.2 * math.sin(2 * math.pi * 440 * i / 16000) + 0.05 * math.sin(
            2 * math.pi * (200 + 1200 * i / 16000) * i / 16000
        )
    return samples


def selected(name: str, tensor: torch.Tensor, row_ids, col_ids) -> dict:
    value = tensor.detach().float().cpu().squeeze(0)
    pairs = []
    for row in row_ids:
        for col in col_ids:
            pairs.append({"row": row, "col": col, "value": float(value[row, col])})
    return {"name": name, "shape": list(value.shape), "samples": pairs}


def load_encoder(model_dir: Path, config: WhisperConfig, dtype: torch.dtype) -> WhisperEncoder:
    encoder = WhisperEncoder(config).to(dtype=dtype).eval()
    checkpoint = model_dir / "model-00000-of-00001.safetensors"
    state = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            if name.startswith(PREFIX):
                state[name[len(PREFIX) :]] = tensors.get_tensor(name).to(dtype=dtype)
    missing, unexpected = encoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch missing={missing} unexpected={unexpected}")
    return encoder


def run_boundaries(encoder: WhisperEncoder, features: np.ndarray, dtype: torch.dtype) -> list[dict]:
    x = torch.from_numpy(features).unsqueeze(0).to(dtype=dtype)
    boundaries = []
    with torch.inference_mode():
        conv1 = torch.nn.functional.gelu(encoder.conv1(x))
        boundaries.append(selected("conv1_gelu", conv1, CHANNELS, FRAMES))
        conv2 = torch.nn.functional.gelu(encoder.conv2(conv1)).permute(0, 2, 1)
        positions = torch.arange(encoder.embed_positions.num_embeddings)
        hidden = conv2 + encoder.embed_positions(positions)
        boundaries.append(selected("conv2_gelu_position", hidden, POSITIONS, DIMS))
        for index, layer in enumerate(encoder.layers):
            hidden = layer(hidden, None, layer_head_mask=None, output_attentions=False)[0]
            boundaries.append(selected(f"layer.{index}", hidden, POSITIONS, DIMS))
        hidden = encoder.layer_norm(hidden)
        boundaries.append(selected("final_layer_norm", hidden, POSITIONS, DIMS))
    return boundaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with (args.model_dir / "config.json").open() as handle:
        root_config = json.load(handle)
    config = WhisperConfig(**root_config["audio_config"])
    extractor = WhisperFeatureExtractor(
        feature_size=80, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400, dither=0.0
    )
    waveform = deterministic_waveform()
    features = extractor._np_extract_fbank_features(np.array([waveform]), "cpu")[0]
    feature_bytes = features.astype("<f4", copy=False).tobytes()

    result = {
        "contract": "moss-whisper-boundaries-v1",
        "transformers": "4.57.1",
        "checkpoint_revision": "0b0295acf3e6282a1692e1f6226faa32a453f7a2",
        "input": {
            "formula": "one_second_440hz_plus_linear_chirp_then_zero_pad_30s",
            "feature_shape": list(features.shape),
            "feature_sha256": hashlib.sha256(feature_bytes).hexdigest(),
        },
        "compute": {},
    }
    # Actual upstream execution and a widened-BF16 diagnostic isolate dtype-boundary drift.
    for label, dtype in (("bf16", torch.bfloat16), ("f32_widened", torch.float32)):
        encoder = load_encoder(args.model_dir, config, dtype)
        result["compute"][label] = run_boundaries(encoder, features, dtype)
        del encoder

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
