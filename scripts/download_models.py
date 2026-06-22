#!/usr/bin/env python3
"""Download go-pherence model assets from Hugging Face.

The script intentionally keeps downloaded files under models/ (ignored by git)
and records the source repo in .huggingface_model for later auditing.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    group: str
    note: str = ""


MODELS: tuple[ModelSpec, ...] = (
    # Tiny/dev models.
    ModelSpec("smollm2-135m", "HuggingFaceTB/SmolLM2-135M", "small", "tiny smoke model"),
    ModelSpec("gemma3-1b-bf16", "google/gemma-3-1b-it", "small", "Gemma3 BF16 smoke"),
    ModelSpec("gemma3-1b-mlx4", "mlx-community/gemma-3-1b-it-4bit", "small", "Gemma3 MLX 4-bit smoke"),
    ModelSpec("qwen2.5-0.5b-mlx4", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "small", "Qwen2.5 MLX 4-bit smoke"),
    ModelSpec("qwen3-0.6b-bf16", "Qwen/Qwen3-0.6B", "small", "Qwen3 BF16 smoke"),
    ModelSpec("qwen3-0.6b-mlx4", "mlx-community/Qwen3-0.6B-4bit", "small", "Qwen3 MLX 4-bit smoke"),

    # Qwen larger/stress assets.
    ModelSpec("qwen2.5-7b-int4", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "qwen", "Qwen2.5 GPTQ int4"),
    ModelSpec("qwen2.5-7b-mlx4", "mlx-community/Qwen2.5-7B-Instruct-4bit", "qwen", "Qwen2.5 MLX 4-bit"),
    ModelSpec("qwen3-30b-a3b-mlx4", "mlx-community/Qwen3-30B-A3B-4bit", "qwen", "Qwen3 MoE MLX 4-bit"),
    ModelSpec("qwen3.6-27b-mlx4-mtp", "samwang0041/Qwen3.6-27B-MLX-4bit-MTP", "qwen", "Qwen3.6 MLX 4-bit native MTP"),
    ModelSpec("qwen3.6-27b-text-nvfp4-mtp", "Qwen/Qwen3.6-27B-Text-NVFP4-MTP", "qwen", "Qwen3.6 NVFP4/native MTP stress asset"),

    # Qwen3-TTS multi-stage speech synthesis checkpoints. Start with the 0.6B
    # CustomVoice model for metadata/prompt inspection and eventual CPU parity.
    ModelSpec("qwen3-tts-0.6b-base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen3tts", "Qwen3-TTS 0.6B Base/reference-audio checkpoint"),
    ModelSpec("qwen3-tts-0.6b-customvoice", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "qwen3tts", "Qwen3-TTS 0.6B CustomVoice first target"),
    ModelSpec("qwen3-tts-1.7b-base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "qwen3tts", "Qwen3-TTS 1.7B Base/reference-audio checkpoint"),
    ModelSpec("qwen3-tts-1.7b-customvoice", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "qwen3tts", "Qwen3-TTS 1.7B CustomVoice checkpoint"),
    ModelSpec("qwen3-tts-1.7b-voicedesign", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "qwen3tts", "Qwen3-TTS 1.7B VoiceDesign checkpoint"),

    # LFM2 hybrid conv/full-attention MoE checkpoint for metadata and future CPU parity work.
    ModelSpec("lfm2.5-8b-a1b", "LiquidAI/LFM2.5-8B-A1B", "lfm2", "LFM2.5 hybrid conv/full-attention MoE"),

    # OpenBMB MiniCPM-V/MiniCPM-O vision-language checkpoints. Some upstream
    # repos are gated and require HUGGINGFACE_TOKEN/HF_TOKEN; keep downloads opt-in.
    ModelSpec("minicpm-v-2.6", "openbmb/MiniCPM-V-2_6", "minicpmv", "MiniCPM-V 2.6 vision-language checkpoint"),
    ModelSpec("minicpm-v-2.0", "openbmb/MiniCPM-V-2", "minicpmv", "MiniCPM-V 2.0 OmniLMM checkpoint"),
    ModelSpec("minicpm-o-2.6", "openbmb/MiniCPM-o-2_6", "minicpmo", "MiniCPM-O 2.6 omni vision-language checkpoint"),

    # Speaker embedding assets for diarization conversion. These are source
    # checkpoints; run scripts/convert_speechbrain_ecapa.py before using them
    # with cmd/diarize-vtt -speaker-model.
    ModelSpec("speechbrain-ecapa-voxceleb", "speechbrain/spkrec-ecapa-voxceleb", "speaker", "SpeechBrain ECAPA speaker verification checkpoint"),

    # Gemma4 assets used by MTP work. These IDs mirror the public MLX naming scheme;
    # if a repo is renamed upstream, override with --repo name=repo or edit this file.
    ModelSpec("gemma4-e2b-it-4bit", "mlx-community/gemma-4-E2B-it-4bit", "gemma4", "Gemma4 E2B verifier"),
    ModelSpec("gemma4-e2b-mlx4", "mlx-community/gemma-4-E2B-it-4bit", "gemma4", "legacy/local alias for E2B MLX4"),
    ModelSpec("gemma4-e2b-mtp-drafter", "mlx-community/gemma-4-E2B-it-assistant-bf16", "gemma4", "Gemma4 E2B MTP drafter"),
    ModelSpec("gemma4-e4b-it-4bit", "mlx-community/gemma-4-E4B-it-4bit", "gemma4", "Gemma4 E4B verifier"),
    ModelSpec("gemma4-e4b-mtp-drafter", "mlx-community/gemma-4-E4B-it-assistant-bf16", "gemma4", "Gemma4 E4B MTP drafter"),
    ModelSpec("gemma4-31b-it-4bit", "mlx-community/gemma-4-31B-it-4bit", "gemma4", "Gemma4 31B stress verifier"),
    ModelSpec("gemma4-31b-it-mtp-assistant-4bit", "mlx-community/gemma-4-31B-it-assistant-4bit", "gemma4", "Gemma4 31B MTP assistant"),
)


def parse_repo_overrides(values: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--repo expects name=repo, got {value!r}")
        name, repo = value.split("=", 1)
        if not name or not repo:
            raise SystemExit(f"--repo expects name=repo, got {value!r}")
        out[name] = repo
    return out


def selected_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = list(MODELS)
    if args.group:
        groups = set(args.group)
        specs = [s for s in specs if s.group in groups]
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            raise SystemExit(f"unknown model(s): {', '.join(sorted(missing))}")
    overrides = parse_repo_overrides(args.repo)
    if overrides:
        specs = [ModelSpec(s.name, overrides.get(s.name, s.repo), s.group, s.note) for s in specs]
    return specs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models", help="destination models directory")
    parser.add_argument("--group", action="append", choices=sorted({s.group for s in MODELS}), help="download only a group; repeatable")
    parser.add_argument("--only", action="append", help="download only this local model name; repeatable")
    parser.add_argument("--repo", action="append", default=[], help="override repo as local_name=org/repo; repeatable")
    parser.add_argument("--revision", default=None, help="optional Hugging Face revision")
    parser.add_argument("--dry-run", action="store_true", help="list selected downloads without downloading")
    parser.add_argument("--force", action="store_true", help="download even if destination already exists")
    parser.add_argument("--local-files-only", action="store_true", help="only use local HF cache")
    parser.add_argument("--allow-pattern", action="append", default=[], help="snapshot_download allow_pattern; repeatable")
    parser.add_argument("--ignore-pattern", action="append", default=[], help="snapshot_download ignore_pattern; repeatable")
    args = parser.parse_args(argv)

    specs = selected_specs(args)
    if not specs:
        raise SystemExit("no models selected")

    root = Path(args.models_dir)
    for s in specs:
        print(f"{s.name:36s} <- {s.repo} [{s.group}] {s.note}")
    if args.dry_run:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: missing huggingface_hub. Install with: python3 -m pip install huggingface_hub", file=sys.stderr)
        return 2

    root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    for s in specs:
        dest = root / s.name
        marker = dest / ".huggingface_model"
        if dest.exists() and marker.exists() and not args.force:
            print(f"skip {s.name}: exists ({marker.read_text().strip()})")
            continue
        print(f"download {s.repo} -> {dest}")
        snapshot_download(
            repo_id=s.repo,
            revision=args.revision,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
            local_files_only=args.local_files_only,
            allow_patterns=args.allow_pattern or None,
            ignore_patterns=args.ignore_pattern or None,
        )
        marker.write_text(s.repo + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
