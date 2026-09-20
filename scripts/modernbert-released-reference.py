#!/usr/bin/env python3
"""CPU-only released ModernBERT-large hidden-state oracle."""
import argparse,json
from pathlib import Path
import torch
from transformers import ModernBertModel
PIN='45bb4654a4d5aaff24dd11d4781fa46d39bf8c13'
def payload(x):a=x.detach().float().cpu().contiguous();return {'shape':list(a.shape),'data':a.reshape(-1).tolist()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if torch.cuda.is_available():raise SystemExit('CUDA must be hidden');torch.set_num_threads(1)
 m=ModernBertModel.from_pretrained(a.model,local_files_only=True,attn_implementation='eager').eval();ids=torch.tensor([[50281,14177,50282]]);mask=torch.ones_like(ids);seen={};hooks=[]
 for i in (0,13,27):hooks.append(m.layers[i].register_forward_hook(lambda mod,args,out,i=i:seen.__setitem__(str(i),out.detach().clone())))
 with torch.no_grad():out=m(input_ids=ids,attention_mask=mask)
 for h in hooks:h.remove()
 data={'model_pin':PIN,'tokens':ids[0].tolist(),'attention_mask':mask[0].tolist(),'layers':{k:payload(v[0]) for k,v in seen.items()},'last_hidden_state':payload(out.last_hidden_state[0])}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n');print('wrote',a.output,a.output.stat().st_size)
if __name__=='__main__':main()
