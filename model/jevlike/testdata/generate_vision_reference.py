"""PYTHONPATH=jevlike-upstream python generate_vision_reference.py"""
import json
from pathlib import Path
import torch
from jevlike.vision import DoomScorerV2

torch.manual_seed(63)
m=DoomScorerV2(width=4,rank=4).float().eval()
# Simple reproducible byte pattern without storing a 76,800-byte observation.
x=(torch.arange(4*120*160,dtype=torch.int64)%251).float().reshape(1,4,120,160)/255
with torch.no_grad():
 features,positions=m.encode(x)
 weights={k:{'shape':list(v.shape),'values':v.flatten().tolist()} for k,v in m.state_dict().items() if k.startswith(('stem.','patch.')) or k=='positions'}
 data={'width':4,'weights':weights,'features':features.tolist(),'positions':positions[0].tolist(),'torch_version':torch.__version__}
Path(__file__).with_name('vision_reference.json').write_text(json.dumps(data,indent=2)+'\n')
