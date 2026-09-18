#!/usr/bin/env python3
"""Small full-backbone fixture using upstream OmniVoice.forward; no downloads."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig
from transformers import Qwen3Config

torch.manual_seed(246)
torch.set_num_threads(2)
root=Path('testdata/omnivoice/backbone')
root.mkdir(parents=True,exist_ok=True)
llm=Qwen3Config(hidden_size=8,intermediate_size=12,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4,vocab_size=32,rms_norm_eps=1e-6,max_position_embeddings=64,rope_parameters={'rope_type':'default','rope_theta':10000.0})
config=OmniVoiceConfig(llm_config=llm.to_dict(),audio_vocab_size=5,audio_mask_id=4,num_audio_codebook=2,audio_codebook_weights=[2,1])
model=OmniVoice(config).eval()
model.llm.config._attn_implementation='eager'
with torch.no_grad():
    for name,p in model.named_parameters():
        p.copy_(1+torch.randn_like(p)*.15 if 'norm' in name else torch.randn_like(p)*.15)
# Config metadata includes the architecture identifiers expected by Go loader.
raw=config.to_dict();raw['architectures']=['OmniVoice'];raw['llm_config']['architectures']=['Qwen3ForCausalLM']
(root/'config.json').write_text(json.dumps(raw,indent=2)+'\n')
state={k:v.detach().contiguous().clone() for k,v in model.state_dict().items()}
save_file(state,str(root/'model.safetensors'))
ids=torch.tensor([[[3,1,4],[0,2,4]]],dtype=torch.long)
audio_mask=torch.tensor([[False,True,True]])
attention_mask=torch.zeros(1,1,3,3)
with torch.inference_mode():
    out=model(input_ids=ids,audio_mask=audio_mask,attention_mask=attention_mask).logits
payload={'tokens':3,'ids':ids.flatten().tolist(),'audio_mask':audio_mask.flatten().tolist(),'logits':out.flatten().tolist(),'shape':list(out.shape),'generator':'upstream OmniVoice.forward; float32 eager attention; deterministic tiny model'}
(root/'expected.json').write_text(json.dumps(payload,indent=2)+'\n')
# Deterministic multi-step reference without random draws; actual upstream
# forward and scoring helpers, with schedule/update loop from _decode.
steps=4;target=2;guidance=2.;penalty=5.
from omnivoice.models.omnivoice import _get_time_steps
times=_get_time_steps(0.,1.,steps,.1)
remaining=target*2;schedule=[]
import math
for step in range(steps):
    k=remaining if step==steps-1 else min(remaining,int(math.ceil((times[step+1]-times[step]).item()*target*2)))
    schedule.append(k);remaining-=k
output=torch.full((2,target),4,dtype=torch.long)
cond=ids.clone();cond[0,:,-target:]=4
uncond=torch.full((1,2,target),4,dtype=torch.long)
with torch.inference_mode():
    for k in schedule:
        c=model(input_ids=cond,audio_mask=audio_mask,attention_mask=torch.zeros(1,1,3,3)).logits[0,:,-target:,:]
        u=model(input_ids=uncond,audio_mask=torch.ones(1,target,dtype=torch.bool),attention_mask=torch.zeros(1,1,target,target)).logits[0]
        if k<=0:continue
        import torch.nn.functional as F
        cp,up=F.log_softmax(c,dim=-1),F.log_softmax(u,dim=-1)
        lp=F.log_softmax(cp+guidance*(cp-up),dim=-1);lp[...,4]=-torch.inf
        pred=lp.argmax(-1);confidence=lp.max(-1)[0]
        scores=confidence-torch.arange(2).unsqueeze(-1)*penalty
        scores=scores.masked_fill(output!=4,-torch.inf)
        chosen=scores.flatten().topk(k).indices
        output.flatten()[chosen]=pred.flatten()[chosen]
        cond[0,:,-target:]=output;uncond[0]=output
(root/'generation.json').write_text(json.dumps({'steps':steps,'target':target,'ids':ids.flatten().tolist(),'audio_mask':audio_mask.flatten().tolist(),'output':output.flatten().tolist(),'schedule':schedule},indent=2)+'\n')
print(out.shape)
