#!/usr/bin/env python3
"""Generate real decoder output fixture, no reference voice/copyrighted audio."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModel
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True,type=Path)
a=p.parse_args();torch.set_num_threads(2)
model=AutoModel.from_pretrained(str(a.model),local_files_only=True,dtype=torch.float32).eval()
codes=torch.tensor([[(book*2+t)*13%1024 for t in range(2)] for book in range(8)],dtype=torch.long).unsqueeze(0)
with torch.inference_mode():
    output=model.decode(codes).audio_values.flatten()
path=Path('testdata/omnivoice/codec-real.json')
path.write_text(json.dumps({'model':'k2-fsa/OmniVoice audio_tokenizer','input_codes':codes.flatten().tolist(),'books':8,'frames':2,'waveform':output.tolist(),'generator':'transformers HiggsAudioV2TokenizerModel.decode, local F16 weights loaded float32','sample_rate':24000},indent=2)+'\n')
print(path,len(output),output[:8])
