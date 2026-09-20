#!/usr/bin/env python3
"""CPU-only tiny Laya decision-head fixture on the pinned ModernBERT contract."""
import argparse,importlib.util,json,sys
from pathlib import Path
import torch
from transformers import ModernBertConfig,ModernBertModel
L='42626c348753fbb17572a813127df2278a1ec527';M='c587bc884db2c2e31fc2b8102314656b17aa07b1'
def payload(x):a=x.detach().float().cpu().contiguous();return {'shape':list(a.shape),'data':a.reshape(-1).tolist()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--upstream',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();sys.path.insert(0,str(a.upstream));from laya.common import DecisionModel
 if torch.cuda.is_available():raise SystemExit('CUDA hidden required');torch.set_num_threads(1);torch.manual_seed(13)
 cfg=ModernBertConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=3,num_attention_heads=4,max_position_embeddings=16,global_attn_every_n_layers=2,local_attention=4,attention_dropout=0,embedding_dropout=0,mlp_dropout=0,attention_bias=False,mlp_bias=False,norm_bias=False,norm_eps=1e-5,pad_token_id=0,_attn_implementation='eager')
 model=DecisionModel(ModernBertModel(cfg),head_layers=2,n_act=2,dropout=0).eval()
 with torch.no_grad():
  for name,v in model.named_parameters():
   if name.endswith('norm.weight') or name.endswith('norm1.weight') or name.endswith('norm2.weight'):v.copy_(1+torch.sin(torch.arange(v.numel()).reshape(v.shape)+len(name))*.02)
   else:v.copy_(torch.sin(torch.arange(v.numel()).reshape(v.shape)*.31+len(name))*.035)
  model.temperature.copy_(torch.tensor([.8,1.1,1.4]))
 ids=torch.tensor([[2,7,4,9,3,0]]);mask=torch.tensor([[1,1,1,1,1,0]]);markers=torch.tensor([[1,3,4]]);mmask=torch.tensor([[1,1,1]],dtype=torch.bool);qtype=torch.tensor([0]);
 with torch.no_grad():logits,actions=model(ids,mask,markers,mmask,qtype)
 data={'laya_pin':L,'transformers_pin':M,'encoder_config':cfg.to_dict(),'tokens':ids[0].tolist(),'attention_mask':mask[0].tolist(),'marker_pos':markers[0].tolist(),'marker_mask':mmask[0].tolist(),'qtype':0,'tensors':{k:payload(v) for k,v in sorted(model.state_dict().items())},'logits':payload(logits[0]),'action_logits':payload(actions[0])}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2)+'\n');print('wrote',a.output,a.output.stat().st_size,len(data['tensors']))
if __name__=='__main__':main()
