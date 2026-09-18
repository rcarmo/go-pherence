#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import HiggsAudioV2TokenizerModel


def build_wave(length: int) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32) / 24000.0
    wave = 0.35 * torch.sin(2 * math.pi * 220.0 * t)
    wave += 0.10 * torch.cos(2 * math.pi * 440.0 * t)
    wave += 0.05 * torch.sin(2 * math.pi * 660.0 * t + 0.3)
    return wave


def build_semantic(frames: int) -> torch.Tensor:
    idx = torch.arange(frames * 768, dtype=torch.float32).reshape(frames, 768)
    semantic = 0.50 * torch.sin(idx * 0.017)
    semantic += 0.25 * torch.cos(idx * 0.011 + 0.2)
    semantic += 0.10 * torch.sin(idx * 0.007 + 0.5)
    return semantic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    model = HiggsAudioV2TokenizerModel.from_pretrained(args.model, local_files_only=True).eval()
    wave = build_wave(args.samples)
    semantic = build_semantic(args.frames)

    with torch.no_grad():
        e_semantic = model.encoder_semantic(semantic.unsqueeze(0).transpose(1, 2))
        acoustic_in = wave.view(1, 1, -1)
        if model._get_conv1d_output_lengths(acoustic_in.shape[-1], model.acoustic_encoder) != e_semantic.shape[-1]:
            acoustic_in = F.pad(acoustic_in, (model.pad, model.pad))
        e_acoustic = model.acoustic_encoder(acoustic_in)
        embeddings = torch.cat([e_acoustic.to(e_semantic.device), e_semantic], dim=1)
        embeddings = model.fc(embeddings.transpose(1, 2)).transpose(1, 2)
        codes = model.quantizer.encode(embeddings).transpose(0, 1)[0]

    fixture = {
        "waveform": wave.tolist(),
        "semantic_frames": int(args.frames),
        "semantic": semantic.reshape(-1).tolist(),
        "codes": codes.reshape(-1).to(torch.int64).tolist(),
        "books": int(codes.shape[0]),
        "frames": int(codes.shape[1]),
    }
    out = Path(args.out)
    out.write_text(json.dumps(fixture), encoding="utf-8")


if __name__ == "__main__":
    main()
