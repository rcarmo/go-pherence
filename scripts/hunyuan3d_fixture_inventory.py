#!/usr/bin/env python3
"""Generate a metadata fixture for a Hunyuan3D shape checkpoint.

This intentionally avoids downloading full model weights. It fetches the YAML
config and, when requested, only the safetensors header bytes needed for tensor
name/shape/dtype inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HF_API = "https://huggingface.co/api/models/{repo}"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"


def fetch_bytes(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "go-pherence-hunyuan3d-inventory/1"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def resolve_url(repo: str, path: str) -> str:
    return HF_RESOLVE.format(repo=repo, path=urllib.parse.quote(path))


def tensor_group(name: str) -> str:
    first = name.split(".", 1)[0]
    if first in {"model", "vae", "conditioner"}:
        return first
    return "other"


def summarize_tensors(tensors: dict[str, Any], max_examples: int = 8) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for name, info in sorted(tensors.items()):
        if name == "__metadata__":
            continue
        group = tensor_group(name)
        g = groups.setdefault(group, {"count": 0, "examples": []})
        g["count"] += 1
        if len(g["examples"]) < max_examples:
            g["examples"].append({
                "name": name,
                "dtype": info.get("dtype"),
                "shape": info.get("shape"),
            })
    return groups


def fetch_safetensors_header(repo: str, path: str) -> dict[str, Any]:
    url = resolve_url(repo, path)
    prefix = fetch_bytes(url, (0, 7))
    if len(prefix) != 8:
        raise RuntimeError(f"{path}: expected 8-byte safetensors prefix, got {len(prefix)}")
    header_len = struct.unpack("<Q", prefix)[0]
    if header_len <= 0 or header_len > 512 * 1024 * 1024:
        raise RuntimeError(f"{path}: unreasonable safetensors header length {header_len}")
    header = fetch_bytes(url, (8, 8 + header_len - 1))
    if len(header) != header_len:
        raise RuntimeError(f"{path}: expected header length {header_len}, got {len(header)}")
    return json.loads(header.decode("utf-8"))


def yaml_scalar(text: str, key: str, default: Any = None) -> Any:
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([^#\n]+)", text)
    if not match:
        return default
    raw = match.group(1).strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if any(c in raw for c in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip('"\'')


def scheduler_reference(config_text: str, steps: int) -> dict[str, Any]:
    if steps <= 0:
        return {}
    train_steps = int(yaml_scalar(config_text, "num_train_timesteps", 1000))
    shift = float(yaml_scalar(config_text, "shift", 1.0))
    use_dynamic = bool(yaml_scalar(config_text, "use_dynamic_shifting", False))
    if steps == 1:
        base_sigmas = [0.0]
    else:
        base_sigmas = [i / (steps - 1) for i in range(steps)]
    if use_dynamic:
        sigmas = base_sigmas
    else:
        sigmas = [shift * s / (1 + (shift - 1) * s) if s != 0 else 0.0 for s in base_sigmas]
    timesteps = [s * train_steps for s in sigmas]
    return {
        "num_inference_steps": steps,
        "num_train_timesteps": train_steps,
        "shift": shift,
        "use_dynamic_shifting": use_dynamic,
        "base_sigmas": base_sigmas,
        "sigmas": sigmas,
        "scheduler_sigmas_with_terminal_one": sigmas + [1.0],
        "timesteps": timesteps,
        "model_timestep_inputs": [t / train_steps for t in timesteps],
        "step_formula": "prev_sample = sample + (sigma_next - sigma) * model_output",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="tencent/Hunyuan3D-2mini", help="Hugging Face repo id")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mini", help="shape model subfolder")
    parser.add_argument("--out", default="testdata/hunyuan3d/fixture-inventory.json", help="output JSON path")
    parser.add_argument("--include-tensors", action="store_true", help="fetch safetensors headers for tensor inventory")
    parser.add_argument("--max-tensor-files", type=int, default=4, help="safety cap for safetensors header reads")
    parser.add_argument("--scheduler-steps", type=int, default=5, help="emit a small FlowMatch scheduler reference")
    args = parser.parse_args()

    repo = args.repo
    subfolder = args.subfolder.strip("/")
    api = fetch_json(HF_API.format(repo=urllib.parse.quote(repo, safe="/")))
    siblings = [s.get("rfilename", "") for s in api.get("siblings", [])]
    files = sorted(p for p in siblings if p == f"{subfolder}/config.yaml" or p.startswith(f"{subfolder}/"))
    config_path = f"{subfolder}/config.yaml"
    if config_path not in files:
        raise SystemExit(f"config not found in repo listing: {config_path}")

    config_bytes = fetch_bytes(resolve_url(repo, config_path))
    config_text = config_bytes.decode("utf-8")
    safetensors_files = [p for p in files if p.endswith(".safetensors")]

    tensor_headers = []
    if args.include_tensors:
        for path in safetensors_files[: args.max_tensor_files]:
            header = fetch_safetensors_header(repo, path)
            tensor_headers.append({
                "path": path,
                "sha256_header": hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest(),
                "groups": summarize_tensors(header),
                "tensor_count": sum(1 for k in header if k != "__metadata__"),
            })

    fixture = {
        "schema": "go-pherence-hunyuan3d-inventory-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo,
        "subfolder": subfolder,
        "config": {
            "path": config_path,
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "bytes": len(config_bytes),
            "text": config_text,
        },
        "scheduler_reference": scheduler_reference(config_text, args.scheduler_steps),
        "files": files,
        "safetensors_files": safetensors_files,
        "tensor_headers": tensor_headers,
        "decision_notes": [
            "Hunyuan3D-2mini standard is the first recommended target because it is smaller than full Hunyuan3D-2 and avoids adding turbo/FlashVDM assumptions to the initial parity path.",
            "This fixture is metadata-only unless --include-tensors is used; even then it fetches only safetensors headers, not tensor payloads.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(files)} files, {len(safetensors_files)} safetensors files, {len(tensor_headers)} tensor headers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
