#!/usr/bin/env python3
"""MiniCPM-V/O scaffold validation smoke for go-pherence."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def write_tiny_safetensors(path: Path) -> None:
    tensors: dict[str, dict[str, object]] = {}
    offset = 0
    specs = {
        "llm.model.embed_tokens.weight": [100, 4],
        "llm.model.layers.0.self_attn.q_proj.weight": [4, 4],
        "vpm.embeddings.patch_embedding.weight": [3, 3, 14, 14],
        "resampler.query.weight": [1, 4],
        "resampler.kv_proj.weight": [4, 3],
    }
    for name, shape in specs.items():
        n = 1
        for dim in shape:
            n *= dim
        byte_len = n * 4
        tensors[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + byte_len]}
        offset += byte_len
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + (b"\x00" * offset))


def write_tiny_tensor_model(root: Path) -> Path:
    (root / "config.json").write_text(json.dumps({
        "architectures": ["MiniCPMVForCausalLM"],
        "model_type": "minicpmv",
        "text_config": {
            "model_type": "qwen2",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 100,
        },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 3,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "image_size": 14,
            "patch_size": 14,
        },
        "resampler_config": {"num_query": 1, "num_heads": 1, "kv_dim": 3},
    }), encoding="utf-8")
    st = root / "tiny.safetensors"
    write_tiny_safetensors(st)
    return st


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
            {"id": 151650, "content": "<audio>"},
            {"id": 151651, "content": "<audio_start>"},
            {"id": 151652, "content": "<audio_end>"},
            {"id": 151653, "content": "<audio_patch>"},
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
    run(["go", "run", "./cmd/minicpmvinspect", "-version"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-version", "-json"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-support-summary"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-support-summary", "-json"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-pending-runtime-steps"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-pending-runtime-steps", "-json"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-capabilities", "-require-capabilities-ready"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-capabilities", "-json", "-require-capabilities-ready"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-fixture-path"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-fixture-path", "-json"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-fixture-summary"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-fixture-summary", "-json"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-require-fixture-ready"], repo)
    run(["go", "run", "./cmd/minicpmvinspect", "-require-fixture-ready", "-json"], repo)
    run([sys.executable, "scripts/download_models.py", "--group", "minicpmv", "--group", "minicpmo", "--dry-run"], repo)
    with tempfile.TemporaryDirectory(prefix="minicpmv-check-") as td:
        model_dir = Path(td)
        write_synthetic_model(model_dir)
        run(["go", "run", "./cmd/minicpmvinspect", "-model", str(model_dir), "-require-config-ready", "-require-metadata-ready"], repo)
    with tempfile.TemporaryDirectory(prefix="minicpmv-tensors-") as td:
        model_dir = Path(td)
        safetensors_path = write_tiny_tensor_model(model_dir)
        run(["go", "run", "./cmd/minicpmvinspect", "-model", str(model_dir), "-safetensors", str(safetensors_path), "-require-tensors-ready", "-require-shapes-ready"], repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
