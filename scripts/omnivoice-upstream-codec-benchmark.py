#!/usr/bin/env python3
"""Upstream decode-only timing, same deterministic codes as Go codec benchmark."""
import argparse
import json
import os
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from transformers import AutoModel
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', required=True, help='audio_tokenizer directory')
p.add_argument('--output', required=True)
a = p.parse_args()
if Path(a.output).exists():
    raise FileExistsError(a.output)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
start = time.perf_counter()
m = AutoModel.from_pretrained(a.model, local_files_only=True, dtype=torch.float32).eval()
load_seconds = time.perf_counter()-start
codes = torch.tensor([[((book*75+t)*13)%1024 for t in range(75)] for book in range(8)], dtype=torch.long).unsqueeze(0)
times = []
with torch.inference_mode():
    m.decode(codes)  # Untimed warm-up, as in Go.
    for _ in range(3):
        start = time.perf_counter()
        wave = m.decode(codes).audio_values.flatten()
        times.append(time.perf_counter()-start)
report = {'torch': torch.__version__, 'threads': 2, 'interop_threads': 1,
          'frames': 75, 'books': 8, 'code_formula': '((book*75+t)*13)%1024',
          'load_seconds_excluded': load_seconds, 'decode_seconds': times,
          'samples': wave.numel(), 'finite': bool(torch.isfinite(wave).all()),
          'timing': 'one untimed warmup, three decode-only calls; no IO/postprocessing'}
with open(a.output, 'x') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
print(json.dumps(report))
