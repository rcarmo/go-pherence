#!/usr/bin/env python3
"""Convert a SpeechBrain ECAPA checkpoint to go-pherence safetensors names.

This is an optional development-time bridge for speaker diarization. Production
Go code only reads the converted safetensors via models/speaker.LoadECAPASafetensors.

Example after downloading speechbrain/spkrec-ecapa-voxceleb assets:

    python3 scripts/convert_speechbrain_ecapa.py \
      --checkpoint /path/to/embedding_model.ckpt \
      --output models/speaker-ecapa-voxceleb.safetensors \
      --dump-keys

The exact SpeechBrain key layout can vary by release. Use --dump-keys first,
then add/adjust --map entries if the default aliases do not match:

    --map conv0.weight=mods.compute_features.conv.weight

Default output tensor names match models/speaker/load.go.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path


CANONICAL_KEYS = [
    "conv0.weight",
    "conv0.bias",
    "pool.attn.weight",
    "pool.attn.bias",
    "pool.out.weight",
    "pool.out.bias",
    "embed.weight",
    "embed.bias",
]

# Best-effort aliases for common SpeechBrain ECAPA-TDNN naming patterns. The
# converter is intentionally strict: if a key is missing, it reports nearby
# checkpoint names instead of guessing silently.
DEFAULT_ALIASES = {
    "conv0.weight": [
        "blocks.0.conv.weight",
        "mods.embedding_model.blocks.0.conv.weight",
        "embedding_model.blocks.0.conv.weight",
    ],
    "conv0.bias": [
        "blocks.0.conv.bias",
        "mods.embedding_model.blocks.0.conv.bias",
        "embedding_model.blocks.0.conv.bias",
    ],
    "pool.attn.weight": [
        "asp.conv.0.weight",
        "mods.embedding_model.asp.conv.0.weight",
        "embedding_model.asp.conv.0.weight",
    ],
    "pool.attn.bias": [
        "asp.conv.0.bias",
        "mods.embedding_model.asp.conv.0.bias",
        "embedding_model.asp.conv.0.bias",
    ],
    "pool.out.weight": [
        "asp.conv.2.weight",
        "mods.embedding_model.asp.conv.2.weight",
        "embedding_model.asp.conv.2.weight",
    ],
    "pool.out.bias": [
        "asp.conv.2.bias",
        "mods.embedding_model.asp.conv.2.bias",
        "embedding_model.asp.conv.2.bias",
    ],
    "embed.weight": [
        "fc.weight",
        "mods.embedding_model.fc.weight",
        "embedding_model.fc.weight",
    ],
    "embed.bias": [
        "fc.bias",
        "mods.embedding_model.fc.bias",
        "embedding_model.fc.bias",
    ],
}


def fail_missing(exc: BaseException) -> None:
    print(
        "missing optional conversion dependency: %s\n"
        "install with: python3 -m pip install torch safetensors" % exc,
        file=sys.stderr,
    )
    raise SystemExit(2)


def parse_map(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --map {value!r}; expected canonical=checkpoint_key")
        left, right = value.split("=", 1)
        out[left.strip()] = right.strip()
    return out


def state_dict_from_checkpoint(obj):
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "embedding_model"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                return nested
            if hasattr(nested, "state_dict"):
                return nested.state_dict()
        return obj
    raise TypeError(f"unsupported checkpoint object {type(obj)!r}")


def choose_key(state: dict, canonical: str, overrides: dict[str, str]) -> str | None:
    if canonical in overrides:
        return overrides[canonical]
    if canonical in state:
        return canonical
    for alias in DEFAULT_ALIASES.get(canonical, []):
        if alias in state:
            return alias
    return None


def nearby_keys(state: dict, canonical: str, limit: int = 12) -> list[str]:
    terms = [part for part in canonical.replace("_", ".").split(".") if part]
    scored = []
    for key in state:
        score = sum(1 for term in terms if term in key.lower())
        if score:
            scored.append((score, key))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [key for _, key in scored[:limit]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="SpeechBrain .ckpt/.pt checkpoint")
    ap.add_argument("--output", required=True, help="Converted safetensors output")
    ap.add_argument("--map", action="append", default=[], help="Override canonical=checkpoint_key mapping; repeatable")
    ap.add_argument("--block", action="append", default=[], help="Map ECAPA block index as go_index:checkpoint_prefix")
    ap.add_argument("--dump-keys", action="store_true", help="Print checkpoint keys before converting")
    args = ap.parse_args()

    try:
        import torch
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - optional script
        fail_missing(exc)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = state_dict_from_checkpoint(ckpt)
    state = OrderedDict((k, v) for k, v in state.items() if hasattr(v, "detach"))

    if args.dump_keys:
        for key, tensor in state.items():
            print(f"{key}\t{tuple(tensor.shape)}\t{tensor.dtype}")

    overrides = parse_map(args.map)
    out = OrderedDict()
    missing = []
    for canonical in CANONICAL_KEYS:
        src = choose_key(state, canonical, overrides)
        if src is None or src not in state:
            missing.append(canonical)
            continue
        out[canonical] = state[src].detach().cpu().contiguous().float()

    for spec in args.block:
        if ":" not in spec:
            raise SystemExit(f"invalid --block {spec!r}; expected go_index:checkpoint_prefix")
        idx, prefix = spec.split(":", 1)
        prefix = prefix.rstrip(".")
        block_map = {
            f"blocks.{idx}.conv.weight": f"{prefix}.conv.weight",
            f"blocks.{idx}.conv.bias": f"{prefix}.conv.bias",
            f"blocks.{idx}.se_down.weight": f"{prefix}.se_down.weight",
            f"blocks.{idx}.se_up.weight": f"{prefix}.se_up.weight",
        }
        for canonical, src in block_map.items():
            if canonical in overrides:
                src = overrides[canonical]
            if src not in state:
                missing.append(canonical)
            else:
                out[canonical] = state[src].detach().cpu().contiguous().float()

    if missing:
        print("missing mappings:", file=sys.stderr)
        for canonical in missing:
            print(f"  {canonical}", file=sys.stderr)
            for key in nearby_keys(state, canonical):
                print(f"    candidate: {key} {tuple(state[key].shape)}", file=sys.stderr)
        print("\nUse --dump-keys and --map/--block to provide exact names.", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(output))
    print(f"wrote {len(out)} tensors to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
