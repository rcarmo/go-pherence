#!/usr/bin/env python3
"""Reference-only complete encoder fixture, synthetic by default; private WAV optional."""
import argparse,json
from pathlib import Path
import torch,torchaudio
from transformers import HiggsAudioV2TokenizerModel
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--wave');p.add_argument('--transcript',default='Synthetic reference.');a=p.parse_args()
torch.set_num_threads(2)
if a.wave:
    import soundfile as sf
    data,sr=sf.read(a.wave,dtype='float32');assert sr==24000
    wave=torch.from_numpy(data);wave=wave.mean(1) if wave.ndim==2 else wave
else:
    t=torch.arange(48000,dtype=torch.float32)/24000
    wave=.07*torch.sin(2*torch.pi*220*t)+.02*torch.sin(2*torch.pi*411*t)
rms=wave.square().mean().sqrt().item()
x=wave*(.1/rms) if 0<rms<.1 else wave
x=x[:x.numel()//960*960]
resampled=torchaudio.functional.resample(x,24000,16000)
m=HiggsAudioV2TokenizerModel.from_pretrained(a.model,local_files_only=True).eval()
with torch.inference_mode():
    codes=m.encode(x[None,None]).audio_codes[0]
Path(a.output).write_text(json.dumps(dict(wave=wave.tolist(),resampled=resampled.tolist(),reference=dict(books=codes.shape[0],frames=codes.shape[1],codes=codes.reshape(-1).tolist(),transcript=a.transcript,ref_rms=rms))))
