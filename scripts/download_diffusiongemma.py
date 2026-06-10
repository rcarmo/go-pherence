#!/usr/bin/env python3
"""Download DiffusionGemma metadata/weights from Hugging Face.

Uses HUGGINGFACE_TOKEN when present. Skips existing files unless --force is set.
"""
import argparse
import json
import os
import pathlib
import shutil
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


def fetch(url: str, dst: pathlib.Path, token: str | None, force: bool, quiet: bool = False) -> None:
    if dst.exists() and not force:
        if not quiet:
            print(f"skip {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "go-pherence-diffusiongemma-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if not quiet:
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
    ap.add_argument("--json-plan", action="store_true", help="emit shard/download plan as JSON")
    ap.add_argument("--ignore-space-check", action="store_true", help="allow full shard download even when free-space preflight is insufficient")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN")
    out = pathlib.Path(args.out)
    base = f"https://huggingface.co/{args.repo}/resolve/main"

    for name in SMALL_FILES:
        fetch(f"{base}/{name}", out / name, token, args.force, args.json_plan)

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
    usage = shutil.disk_usage(out)
    free = usage.free
    present_shards = [s for s in shards if (out / s).exists()]
    missing_shards = [s for s in shards if not (out / s).exists()]
    present_bytes = sum((out / s).stat().st_size for s in present_shards)
    plan = {
        "repo": args.repo,
        "out": str(out),
        "shards": len(shards),
        "shard_files": shards,
        "present_shards": present_shards,
        "missing_shards": missing_shards,
        "present_shard_count": len(present_shards),
        "missing_shard_count": len(missing_shards),
        "present_bytes": present_bytes,
        "present_byte_percent": (100 * present_bytes / total_size) if total_size else 0,
        "total_size_bytes": total_size,
        "total_size_gib": total_size / (1024**3) if total_size else 0,
        "total_parameters": total_params,
        "target_free_bytes": free,
        "target_free_gib": free / (1024**3),
        "enough_space": (free >= total_size) if total_size else None,
    }
    if args.json_plan:
        print(json.dumps(plan, indent=2))
    elif total_size:
        print(f"checkpoint shards={len(shards)} total_size={total_size} bytes ({total_size/(1024**3):.2f} GiB) parameters={total_params}")
        print(f"target_free={free} bytes ({free/(1024**3):.2f} GiB) enough_space={free >= total_size}")
    else:
        print(f"checkpoint shards={len(shards)}")
    if args.plan_only:
        return 0
    if total_size and free < total_size and not args.ignore_space_check:
        print(
            f"refusing shard download: target has {free} bytes ({free/(1024**3):.2f} GiB) free, "
            f"checkpoint requires {total_size} bytes ({total_size/(1024**3):.2f} GiB). "
            "Use --ignore-space-check to override.",
            file=sys.stderr,
        )
        return 3
    for shard in shards:
        fetch(f"{base}/{shard}", out / shard, token, args.force, args.json_plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
