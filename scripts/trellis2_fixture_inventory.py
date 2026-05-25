#!/usr/bin/env python3
"""Generate a compact TRELLIS.2 metadata inventory.

Default mode reads Hugging Face metadata and JSON config files without fetching
safetensors payloads. With --include-tensors, it reads only safetensors header
bytes from local files or HTTP range requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_REPO = "microsoft/TRELLIS.2-4B"
DEFAULT_FILES = [
    "pipeline.json",
    "texturing_pipeline.json",
    "ckpts/shape_enc_next_dc_f16c32_fp16.json",
    "ckpts/shape_dec_next_dc_f16c32_fp16.json",
    "ckpts/tex_enc_next_dc_f16c32_fp16.json",
    "ckpts/tex_dec_next_dc_f16c32_fp16.json",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
]


def hf_raw_url(repo: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def read_url(url: str, timeout: int = 30, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def read_json_from_source(source: Path | str, repo: str, revision: str) -> tuple[Any, str]:
    if isinstance(source, Path):
        data = source.read_bytes()
        label = str(source)
    else:
        url = hf_raw_url(repo, source, revision)
        data = read_url(url)
        label = url
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def checkpoint_family(name: str) -> str:
    base = Path(name).name
    if base.startswith("shape_enc_"):
        return "shape_encoder"
    if base.startswith("shape_dec_"):
        return "shape_decoder"
    if base.startswith("tex_enc_"):
        return "texture_encoder"
    if base.startswith("tex_dec_"):
        return "texture_decoder"
    if base.startswith("ss_flow_"):
        return "sparse_structure_flow"
    if base.startswith("slat_flow_img2shape_"):
        return "structured_latent_shape_flow"
    if base.startswith("slat_flow_imgshape2tex_"):
        return "structured_latent_texture_flow"
    if base == "pipeline.json":
        return "pipeline"
    if base == "texturing_pipeline.json":
        return "texturing_pipeline"
    return "unknown"


def summarize_config(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"type": type(obj).__name__}
    out: dict[str, Any] = {"top_level_keys": sorted(obj.keys())}
    for key in ("model", "models", "sampler", "image_cond_model", "sparse_structure_flow_model", "slat_flow_model", "shape_model", "texture_model"):
        if key in obj:
            out[key] = summarize_value(obj[key])
    for key in ("name", "type", "target", "resolution", "dtype", "num_steps", "num_channels", "hidden_size", "num_heads", "depth"):
        if key in obj:
            out[key] = obj[key]
    return out


def summarize_value(v: Any) -> Any:
    if isinstance(v, dict):
        summary = {k: v[k] for k in ("name", "type", "target", "resolution", "dtype") if k in v}
        summary["keys"] = sorted(v.keys())[:32]
        return summary
    if isinstance(v, list):
        return {"len": len(v), "first": summarize_value(v[0]) if v else None}
    return v


def read_safetensors_header_local(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
    return json.loads(header.decode("utf-8"))


def read_safetensors_header_hf(repo: str, filename: str, revision: str) -> dict[str, Any]:
    url = hf_raw_url(repo, filename, revision)
    first = read_url(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", first[:8])[0]
    data = read_url(url, headers={"Range": f"bytes=8-{7+n}"})
    return json.loads(data[:n].decode("utf-8"))


def summarize_safetensors_header(header: dict[str, Any]) -> dict[str, Any]:
    names = sorted(k for k in header.keys() if k != "__metadata__")
    dtypes: dict[str, int] = {}
    for name in names:
        dtype = header.get(name, {}).get("dtype", "unknown")
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    return {
        "tensor_count": len(names),
        "dtypes": dict(sorted(dtypes.items())),
        "examples": names[:16],
        "metadata": header.get("__metadata__", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--local-dir", default="", help="optional local TRELLIS.2 model snapshot directory")
    ap.add_argument("--out", default="/workspace/tmp/trellis2-inventory.json")
    ap.add_argument("--include-tensors", action="store_true", help="read safetensors headers only")
    args = ap.parse_args()

    local = Path(args.local_dir) if args.local_dir else None
    entries: list[dict[str, Any]] = []
    for rel in DEFAULT_FILES:
        source: Path | str = local / rel if local else rel
        obj, sha = read_json_from_source(source, args.repo, args.revision)
        entry = {
            "path": rel,
            "family": checkpoint_family(rel),
            "sha256": sha,
            "summary": summarize_config(obj),
        }
        st_rel = rel[:-5] + ".safetensors" if rel.endswith(".json") and rel.startswith("ckpts/") else ""
        if args.include_tensors and st_rel:
            try:
                header = read_safetensors_header_local(local / st_rel) if local else read_safetensors_header_hf(args.repo, st_rel, args.revision)
                entry["safetensors"] = summarize_safetensors_header(header)
            except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError, struct.error) as exc:
                entry["safetensors_error"] = str(exc)
        entries.append(entry)

    report = {
        "repo": args.repo,
        "revision": args.revision,
        "local_dir": str(local) if local else "",
        "entries": entries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
