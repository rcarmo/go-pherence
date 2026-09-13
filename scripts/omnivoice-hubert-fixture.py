#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import HiggsAudioV2TokenizerModel


def build_wave(samples: int) -> torch.Tensor:
    t = torch.arange(samples, dtype=torch.float32) / 16000.0
    wave = 0.35 * torch.sin(2 * math.pi * 220.0 * t)
    wave += 0.10 * torch.cos(2 * math.pi * 440.0 * t)
    wave += 0.05 * torch.sin(2 * math.pi * 660.0 * t + 0.3)
    return F.pad(wave, (160, 160))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=1600)
    args = parser.parse_args()

    model = HiggsAudioV2TokenizerModel.from_pretrained(args.model, local_files_only=True).eval()
    wave = build_wave(args.samples)
    with torch.no_grad():
        outputs = model.semantic_model(wave.unsqueeze(0), output_hidden_states=True, return_dict=True)
        semantic = torch.stack([h.float() for h in outputs.hidden_states], dim=1).mean(dim=1)[0]

    fixture = {
        "input": wave.tolist(),
        "frames": int(semantic.shape[0]),
        "semantic": semantic.reshape(-1).tolist(),
    }
    out = Path(args.out)
    out.write_text(json.dumps(fixture), encoding="utf-8")


if __name__ == "__main__":
    main()
