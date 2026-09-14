#!/usr/bin/env python3
"""Compare every logit of a real OmniVoice forward to native Go; not speech."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import torch
from omnivoice import OmniVoice

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True,type=Path)
a=p.parse_args()
os.environ['HF_HUB_OFFLINE']='1'
torch.set_num_threads(2)
cfg=json.loads((a.model/'config.json').read_text());books=cfg['num_audio_codebook']
# Real text id plus two masked audio positions, fixed and shared across runtimes.
ids=[[100, cfg['audio_mask_id'],cfg['audio_mask_id']] for _ in range(books)]
start=time.monotonic()
model=OmniVoice.from_pretrained(str(a.model),device_map='cpu',dtype=torch.float32,load_asr=False,local_files_only=True).eval()
with torch.inference_mode():
    result=model(input_ids=torch.tensor([ids]),audio_mask=torch.tensor([[False,True,True]]),attention_mask=torch.zeros(1,1,3,3)).logits.cpu().flatten().clone()
python_seconds=time.monotonic()-start
del model
gc.collect()
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'input.json'
    path.write_text(json.dumps({'tokens':3,'ids':[x for row in ids for x in row],'audio_mask':[False,True,True]}))
    native=json.loads(subprocess.check_output(['go','run','./cmd/audio/omnivoice','-mode','logits','-model',str(a.model),'-input',str(path)],text=True))
actual=torch.tensor(native.pop('logits'))
diff=(actual-result).abs()
report={'logits_compared':len(actual),'max_absolute_error':diff.max().item(),'mean_absolute_error':diff.mean().item(),'python_load_and_forward_seconds':python_seconds,'native':native}
print(json.dumps(report,indent=2))
# Full 28-layer float32 reduction-order divergence; compare all logits, not
# just argmax. No quantization or altered fixture-dependent tolerance.
assert torch.isfinite(actual).all() and torch.allclose(actual,result,atol=0.003,rtol=0.0003)
