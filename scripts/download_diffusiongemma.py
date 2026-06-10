#!/usr/bin/env python3
"""Download DiffusionGemma metadata/weights from Hugging Face.

Uses HUGGINGFACE_TOKEN when present. Skips existing files unless --force is set.
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.request

DEFAULT_REPO = "google/diffusiongemma-26B-A4B-it"
SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
]


def fetch(url: str, dst: pathlib.Path, token: str | None, force: bool) -> None:
    if dst.exists() and not force:
        print(f"skip {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "go-pherence-diffusiongemma-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(f"download {url} -> {dst}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default="models/diffusiongemma-26B-A4B-it")
    ap.add_argument("--metadata-only", action="store_true", help="download configs/index only, not safetensor shards")
    ap.add_argument("--plan-only", action="store_true", help="download/read metadata, print shard plan, and exit before shard downloads")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN")
    out = pathlib.Path(args.out)
    base = f"https://huggingface.co/{args.repo}/resolve/main"

    for name in SMALL_FILES:
        fetch(f"{base}/{name}", out / name, token, args.force)

    if args.metadata_only and not args.plan_only:
        return 0

    index_path = out / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        print("no shards found in index", file=sys.stderr)
        return 1
    meta = index.get("metadata", {})
    total_size = int(meta.get("total_size") or 0)
    total_params = int(meta.get("total_parameters") or 0)
    if total_size:
        print(f"checkpoint shards={len(shards)} total_size={total_size} bytes ({total_size/(1024**3):.2f} GiB) parameters={total_params}")
    else:
        print(f"checkpoint shards={len(shards)}")
    if args.plan_only:
        return 0
    for shard in shards:
        fetch(f"{base}/{shard}", out / shard, token, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
