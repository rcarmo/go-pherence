#!/usr/bin/env python3
"""MiniCPM-V/O scaffold validation smoke for go-pherence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def write_synthetic_model(root: Path) -> None:
    (root / "config.json").write_text(json.dumps({
        "architectures": ["MiniCPMOForCausalLM"],
        "model_type": "minicpm-o",
        "text_config": {
            "model_type": "qwen2",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
            "vocab_size": 151666,
        },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "image_size": 448,
            "patch_size": 14,
        },
        "audio_config": {
            "model_type": "whisper_encoder",
            "hidden_size": 1280,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "feature_size": 128,
            "num_mel_bins": 128,
            "sampling_rate": 16000,
        },
        "resampler_config": {"num_query": 64, "num_heads": 28, "kv_dim": 1152},
    }), encoding="utf-8")
    (root / "preprocessor_config.json").write_text(json.dumps({
        "image_processor_type": "SiglipImageProcessor",
        "size": {"height": 448, "width": 448},
        "patch_size": 14,
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 1.0 / 255.0,
    }), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Qwen2Tokenizer",
        "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}<image>{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}",
    }), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [
            {"id": 151640, "content": "<im_start>"},
            {"id": 151641, "content": "<im_end>"},
            {"id": 151642, "content": "<im_patch>"},
            {"id": 151643, "content": "<image>"},
        ],
        "model": {"vocab": {}},
    }), encoding="utf-8")
    (root / "generation_config.json").write_text(json.dumps({
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-go-test", action="store_true", help="skip focused go test packages")
    parser.add_argument("--skip-build", action="store_true", help="skip building bin/minicpmvinspect")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if not args.skip_go_test:
        run(["go", "test", "./loader/config", "./model/minicpmv", "./cmd/minicpmvinspect", "-count=1"], repo)
    if not args.skip_build:
        run(["go", "build", "-o", "bin/minicpmvinspect", "./cmd/minicpmvinspect"], repo)
    run([sys.executable, "scripts/download_models.py", "--group", "minicpmv", "--group", "minicpmo", "--dry-run"], repo)
    with tempfile.TemporaryDirectory(prefix="minicpmv-check-") as td:
        model_dir = Path(td)
        write_synthetic_model(model_dir)
        run(["go", "run", "./cmd/minicpmvinspect", "-model", str(model_dir), "-require-config-ready", "-require-metadata-ready"], repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
