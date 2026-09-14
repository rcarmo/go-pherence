import torch, numpy as np, json, gzip
from pathlib import Path
torch.set_num_threads(1)
f=json.loads(gzip.decompress(Path('projects/go-pherence/models/speaker/community1/testdata/sincnet-reference.json.gz').read_bytes()))
lo=50+torch.tensor(f['weights']['LowHz']).abs().view(-1,1); hi=(lo+50+torch.tensor(f['weights']['BandHz']).abs().view(-1,1)).clamp(50,8000)
n=2*np.pi*(torch.arange(-125,0.).view(1,-1)/16000); w=torch.from_numpy(np.hamming(251)[:125]).float()
a=lo@n;b=hi@n
v={'n':n.flatten().tolist(),'window':w.tolist(),'lo':a.flatten().tolist(),'hi':b.flatten().tolist(),'sinlo':a.sin().flatten().tolist(),'sinhi':b.sin().flatten().tolist(),'coslo':a.cos().flatten().tolist(),'coshi':b.cos().flatten().tolist()}
Path('tmp/sincnet-diagnosis/filter.json').write_text(json.dumps(v))
