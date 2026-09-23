#!/usr/bin/env python3
"""Generate exact one-row depth/CFG distillation oracle from pinned upstream."""
import inspect, json, subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from training.modules.conditioner import build_sequences_with_conditions
from training.modules.model import TrainableTTS
from training.scripts.shrink_checkpoint import select_layers
from pocket_tts.modules.transformer import StreamingTransformer

PIN="0acce6b2f390150267557770d2098c5caa9a18ac";root=Path.cwd().resolve()
if subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()!=PIN: raise SystemExit("wrong upstream revision")
for obj,rel in [(build_sequences_with_conditions,"training/modules/conditioner.py"),(TrainableTTS,"training/modules/model.py"),(select_layers,"training/scripts/shrink_checkpoint.py")]:
    if Path(inspect.getfile(obj)).resolve()!=root/rel: raise SystemExit(f"wrong import for {rel}")
subprocess.run(["git","diff","--quiet","HEAD","--","training/modules/conditioner.py","training/modules/model.py","training/scripts/shrink_checkpoint.py","pocket_tts/modules/transformer.py","pocket_tts/modules/attention.py","pocket_tts/modules/rope.py","pocket_tts/modules/layer_scale.py"],cwd=root,check=True)
torch.manual_seed(0)

def fill_linear(linear,start):
    with torch.no_grad():
        for i in range(linear.weight.numel()): linear.weight.flatten()[i]=start+((i*7)%13-6)*.025
        if linear.bias is not None:
            for i in range(linear.bias.numel()): linear.bias[i]=start/3+(i-1)*.02

class Conditioner(nn.Module):
    def __init__(self): super().__init__();self.embed=nn.Embedding(6,4)
    def forward(self,x): return self.embed(x)
class FlowLM(nn.Module):
    def __init__(self,layers,offset):
        super().__init__();self.conditioner=Conditioner();self.bos_emb=nn.Parameter(torch.tensor([.15+offset,-.25-offset]));self.bos_before_voice=nn.Parameter(torch.tensor([[[.05+offset,-.1,.2,-.3-offset]]]));self.speaker_proj_weight=nn.Parameter(torch.empty(4,2));self.input_linear=nn.Linear(2,4,bias=False);self.transformer=StreamingTransformer(d_model=4,num_heads=2,num_layers=layers,layer_scale=1.0,dim_feedforward=6,context=None,max_period=100.);self.out_norm=nn.LayerNorm(4,eps=1e-5);self.out_eos=nn.Linear(4,1)
        with torch.no_grad():
            for i in range(self.conditioner.embed.weight.numel()):self.conditioner.embed.weight.flatten()[i]=offset+((i*5)%17-8)*.03
            for i in range(self.speaker_proj_weight.numel()):self.speaker_proj_weight.flatten()[i]=offset+.02+((i*7)%13-6)*.025
            self.out_norm.weight.copy_(torch.tensor([1.02,.98,1.08,.92])+offset);self.out_norm.bias.copy_(torch.tensor([.01,-.02,.03,-.04])+offset)
        fill_linear(self.input_linear,-.015+offset);fill_linear(self.out_eos,.01+offset)
        for n,layer in enumerate(self.transformer.layers):
            d=offset+n*.007;fill_linear(layer.self_attn.in_proj,.01+d);fill_linear(layer.self_attn.out_proj,-.02+d);fill_linear(layer.linear1,.03+d);fill_linear(layer.linear2,-.015+d)
            with torch.no_grad():
                layer.norm1.weight.copy_(torch.tensor([1.1,.9,1.2,.8])+d);layer.norm1.bias.copy_(torch.tensor([.02,-.03,.01,.04])+d);layer.norm2.weight.copy_(torch.tensor([.95,1.05,.85,1.15])+d);layer.norm2.bias.copy_(torch.tensor([-.01,.02,-.04,.03])+d);layer.layer_scale_1.scale.copy_(torch.tensor([.8,1.1,.9,1.2])+d);layer.layer_scale_2.scale.copy_(torch.tensor([1.05,.95,1.15,.85])+d)

def backbone(model,latents,voice,tokens,force_null=False):
    args=SimpleNamespace(voice_dropout=0.,text_dropout=0.)
    x,prefix=build_sequences_with_conditions(args,latents,tokens,voice,cfg_dropout=False,fl=model,num_voice_prompt_frames=torch.tensor([2]),force_null=force_null)
    out=model.out_norm(model.transformer(x,model_state=None));idx=prefix[:,None]+torch.arange(latents.shape[1])[None,:]
    return out.gather(1,idx[:,:,None].expand(-1,-1,out.shape[-1]))

student,teacher=FlowLM(1,0.),FlowLM(2,.04)
for p in teacher.parameters():p.requires_grad_(False)
latents=torch.tensor([[[.2,-.4],[.6,.1],[-.3,.7]]],requires_grad=True);voice=torch.tensor([[[-.2,.5],[.8,-.1]]],requires_grad=True);tokens=[torch.tensor([1,3])];mask=torch.tensor([[True,False,False]]);cfg=2.
z=backbone(student,latents,voice,tokens)
with torch.no_grad():
    zc=backbone(teacher,latents,voice,tokens)
    zn=backbone(teacher,latents,voice,tokens,True)
    target=zn+cfg*(zc-zn)
shifted=torch.cat([mask[:,:1],mask[:,:-1]],1);loss=((z-target).square().mean(-1)*shifted).sum()/shifted.sum().clamp(min=1);loss.backward()
fixture={"schema":1,"upstream_revision":PIN,"generator":f"pocket-tts .venv torch {torch.__version__}","frames":3,"voice_frames":2,"hidden":4,"latent_dim":2,"vocabulary":6,"cfg_coef":cfg,"mask":mask.flatten().tolist(),"shifted_mask":shifted.flatten().tolist(),"normalized_latents":latents.detach().flatten().tolist(),"voice_latents":voice.detach().flatten().tolist(),"text_tokens":tokens[0].tolist(),"student_layers":1,"teacher_layers":2,"selected_24_to_6":select_layers(24,6),"loss":loss.item(),"student":z.detach().flatten().tolist(),"teacher_conditioned":zc.flatten().tolist(),"teacher_null":zn.flatten().tolist(),"target":target.flatten().tolist(),"d_normalized_latents":latents.grad.flatten().tolist(),"d_voice_latents":voice.grad.flatten().tolist(),"student_parameters":{k:v.detach().flatten().tolist() for k,v in student.state_dict().items()},"teacher_parameters":{k:v.detach().flatten().tolist() for k,v in teacher.state_dict().items()},"student_gradients":{k:v.grad.detach().flatten().tolist() for k,v in student.named_parameters() if v.grad is not None}}
print(json.dumps(fixture,indent=2))
