#!/usr/bin/env python3
"""Freeze the pinned upstream MimiModel.state_dict() names and shapes."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5] / "tmp" / "pocket-tts"
PIN = "0acce6b2f390150267557770d2098c5caa9a18ac"
if subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() != PIN:
    raise SystemExit("pocket-tts checkout is not pinned")
import sys
sys.path.insert(0, str(ROOT))
from pocket_tts.models.mimi import build_mimi
from training.modules.builders import load_model_config
cfg = load_model_config(str(ROOT / "pocket_tts/config/english.yaml"), {})
model = build_mimi(cfg.mimi)
out = {"schema": 1, "upstream_revision": PIN, "config": "pocket_tts/config/english.yaml", "tensors": {f"mimi.{k}": list(v.shape) for k, v in sorted(model.state_dict().items())}}
path = Path(__file__).with_name("mimi_state_shapes_pytorch.json")
path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(path)
