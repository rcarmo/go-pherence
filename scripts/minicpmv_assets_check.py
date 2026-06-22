#!/usr/bin/env python3
"""Discover and inspect local MiniCPM-V/O model directories."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def has_config(path: Path) -> bool:
    return (path / "config.json").is_file()


def discover(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and "minicpm" in path.name.lower() and has_config(path):
            out.append(path)
    return out


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models", help="model root to scan")
    parser.add_argument("--require-assets", action="store_true", help="exit non-zero when no MiniCPM assets are found")
    parser.add_argument("--strict", action="store_true", help="pass -strict to minicpmvinspect for every discovered model")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    models_dir = (repo / args.models_dir).resolve()
    models = discover(models_dir)
    if not models:
        print(f"no MiniCPM-V/O model directories found under {models_dir}")
        return 1 if args.require_assets else 0
    for model in models:
        flags = ["-model", str(model), "-require-config-ready"]
        if args.strict:
            flags.append("-strict")
        run(["go", "run", "./cmd/minicpmvinspect", *flags], repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
