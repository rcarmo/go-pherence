#!/usr/bin/env python3
"""CPU-only tiny ModernBERT hidden-state fixture from pinned Transformers."""
import argparse,json,math
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import ModernBertConfig,ModernBertModel

TRANSFORMERS_PIN='c587bc884db2c2e31fc2b8102314656b17aa07b1'
MODEL_PIN='45bb4654a4d5aaff24dd11d4781fa46d39bf8c13'

def payload(x):
 a=x.detach().to(torch.float32).cpu().contiguous()
 return {'shape':list(a.shape),'data':a.reshape(-1).tolist()}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
 if torch.cuda.is_available():raise SystemExit('CUDA must be hidden for reference generation')
 torch.set_num_threads(1);torch.manual_seed(0)
 cfg=ModernBertConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=3,num_attention_heads=4,max_position_embeddings=16,global_attn_every_n_layers=2,local_attention=4,global_rope_theta=160000.0,local_rope_theta=10000.0,attention_dropout=0.0,embedding_dropout=0.0,mlp_dropout=0.0,attention_bias=False,mlp_bias=False,norm_bias=False,norm_eps=1e-5,pad_token_id=0,_attn_implementation='eager')
 m=ModernBertModel(cfg).eval()
 with torch.no_grad():
  for name,v in m.named_parameters():
   if name.endswith('.weight') and 'norm' in name: v.copy_(1+torch.sin(torch.arange(v.numel(),dtype=torch.float32).reshape(v.shape)+len(name))*.03)
   else:v.copy_(torch.sin(torch.arange(v.numel(),dtype=torch.float32).reshape(v.shape)*.37+len(name))*.04)
 ids=torch.tensor([[2,7,4,9,3]],dtype=torch.long);mask=torch.tensor([[1,1,1,0,0]],dtype=torch.long)
 layer_outputs=[]
 hooks=[layer.register_forward_hook(lambda mod,args,out:layer_outputs.append(out.detach().clone())) for layer in m.layers]
 with torch.no_grad():
  embedded=m.embeddings(ids)
  out=m(input_ids=ids,attention_mask=mask)
 for hook in hooks:hook.remove()
 cfg_json=cfg.to_dict();cfg_json['_attn_implementation']='eager'
 data={'transformers_pin':TRANSFORMERS_PIN,'model_pin':MODEL_PIN,'config':cfg_json,'tokens':ids[0].tolist(),'attention_mask':mask[0].tolist(),'layer_types':cfg.layer_types,'tensors':{k:payload(v) for k,v in sorted(m.state_dict().items())},'hidden_states':[payload(embedded[0])]+[payload(x[0]) for x in layer_outputs],'last_hidden_state':payload(out.last_hidden_state[0])}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2)+'\n');save_file({'model.'+k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},str(args.output.with_suffix('.safetensors')));args.output.with_name('tiny-config.json').write_text(json.dumps(cfg_json,indent=2)+'\n');print('wrote',args.output,args.output.stat().st_size,len(data['tensors']),data['layer_types'])
if __name__=='__main__':main()
