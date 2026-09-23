#!/usr/bin/env python3
"""Generate released raw-waveform Mimi encoder parity from the gated bundle."""
import hashlib, inspect, json, math, subprocess
from pathlib import Path
import torch
import safetensors.torch
from pocket_tts.models.mimi import build_mimi, MimiModel
from training.modules.builders import load_model_config
PIN="0acce6b2f390150267557770d2098c5caa9a18ac";WEIGHTS_REV="983151f13aaeab1b13c1e5e3c2c383d49a9edf3f";WEIGHTS_SHA="fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997"
root=Path.cwd().resolve()
if subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()!=PIN:raise SystemExit("wrong upstream revision")
if Path(inspect.getfile(MimiModel)).resolve()!=root/"pocket_tts/models/mimi.py":raise SystemExit("wrong Mimi import")
subprocess.run(["git","diff","--quiet","HEAD","--","pocket_tts/models/mimi.py","pocket_tts/modules/seanet.py","pocket_tts/modules/conv.py","pocket_tts/modules/resample.py","pocket_tts/modules/transformer.py","pocket_tts/modules/attention.py","pocket_tts/modules/rope.py"],cwd=root,check=True)
weights=Path("/workspace/checkpoints/pocket-tts/languages/english/model.safetensors")
if hashlib.sha256(weights.read_bytes()).hexdigest()!=WEIGHTS_SHA:raise SystemExit("wrong gated weights")
cfg=load_model_config("pocket_tts/config/english.yaml",{});m=build_mimi(cfg.mimi);state=safetensors.torch.load_file(weights);m.load_state_dict({k.removeprefix("mimi."):v for k,v in state.items() if k.startswith("mimi.")},strict=True);m.eval()
n=30721;t=torch.arange(n,dtype=torch.float32);audio=(.18*torch.sin(2*math.pi*220*t/24000)+.07*torch.sin(2*math.pi*731*t/24000)+.02*(2*(t/(n-1)) - 1))[None,None]
checkpoints={}
def hook(name):
 def f(_m,_i,o):
  if isinstance(o,(list,tuple)):o=o[0]
  x=o.detach().float().contiguous().cpu();raw=x.numpy().tobytes();flat=x.flatten();checkpoints[name]={"shape":list(x.shape),"sha256":hashlib.sha256(raw).hexdigest(),"head":flat[:16].tolist(),"tail":flat[-16:].tolist(),"sum":x.sum().item(),"square_sum":x.square().sum().item()}
 return f
handles=[]
for i,layer in enumerate(m.encoder.model):
 if hasattr(layer,"conv") or hasattr(layer,"block"):handles.append(layer.register_forward_hook(hook(f"encoder.model.{i}")))
handles.append(m.encoder_transformer.register_forward_hook(hook("encoder_transformer")));handles.append(m.downsample.register_forward_hook(hook("downsample")))
with torch.no_grad():latent=m.encode_to_latent(audio)
for h in handles:h.remove()
out={"schema":1,"upstream_revision":PIN,"weights_revision":WEIGHTS_REV,"weights_sha256":WEIGHTS_SHA,"sample_rate":24000,"samples":n,"frame_rate":12.5,"latent_shape":list(latent.shape),"audio":audio.flatten().tolist(),"latent":latent.flatten().tolist(),"latent_sha256":hashlib.sha256(latent.float().contiguous().cpu().numpy().tobytes()).hexdigest(),"checkpoints":checkpoints}
fixture=Path(__file__).with_name("mimi_encoder_upstream_latents.safetensors");safetensors.torch.save_file({"latents":latent[0].float().contiguous().cpu()},fixture)
print(json.dumps(out,indent=2,sort_keys=True))
