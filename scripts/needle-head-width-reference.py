#!/usr/bin/env python3
"""CPU-only head-gradient and half-width parity from pinned Needle 3 source.
Objectives are explicit BCE/CE/embedding-MSE, not an upstream dataset recipe.
"""
import argparse
import importlib.util
import json
from pathlib import Path
spec=importlib.util.spec_from_file_location('ref',Path(__file__).with_name('needle-reference.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict,unflatten_dict


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--upstream',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
 pin=r._assert_upstream_pin(args.upstream);a=r._load_architecture(args.upstream);a._quantize.configure_deploy(act_bits=8,kv_bits=8)
 fixture=json.loads((Path(__file__).parent.parent/'model/needle/testdata/needle3-extended.json').read_text())['base']
 params=unflatten_dict({tuple(k.split('/')):jnp.array(v['data'],jnp.float32).reshape(v['shape']) for k,v in fixture['tensors'].items()})
 config=a.TransformerConfig(**fixture['config']);model=a.SimpleAttentionNetwork(config);ids=jnp.array([fixture['tokens']],jnp.int32)
 heads={}
 for kind,target in [('confidence',[1.]),('router',[2.]),('embedding',[1.,0.,0.,0.])]:
  method=getattr(model,'forward_'+kind)
  heads[kind]={}
  for quant in (False,True):
   def loss_fn(p):
    applied=a._quantize.cq_ste_params(p,4) if quant else p
    output=model.apply({'params':applied},ids,method=method,quant=quant)
    if kind=='confidence': loss=jnp.mean(jax.nn.softplus(output)-output*target[0])
    elif kind=='router': loss=-jax.nn.log_softmax(output,axis=-1)[0,int(target[0])]
    else: loss=jnp.mean((output-jnp.array(target,jnp.float32))**2)
    return loss,output
   (loss,output),grads=jax.value_and_grad(loss_fn,has_aux=True)(params)
   heads[kind]['cq' if quant else 'fp32']={'target':target,'loss':float(loss),'output':r._tensor_payload(output),'gradients':r._named_tensor_map(grads)}
 # Tiny no-engram half-width model avoids inventing explicit engram geometry
 # that width_config does not remap. Trained split permutations are enabled.
 cfg=a.TransformerConfig(vocab_size=16,d_model=16,num_heads=4,num_kv_heads=1,qk_head_dim=4,v_head_dim=4,num_layers=2,max_seq_len=16,mhc_lanes=2,qkv_conv_taps=3,engram_layers=(),global_layers=(1,),sliding_window=3,ladder_widths=(8,),dtype='float32',flash=False,remat=False,embedding_dim=4,embedding_probes=2,embedding_queries=2,confidence_probes=2,confidence_queries=2,router_probes=2,router_queries=2)
 m=a.SimpleAttentionNetwork(cfg);tokens=jnp.array([[2,7,4,9,3]],jnp.int32)
 p=m.init(jax.random.key(7),tokens)['params']
 for i,name in enumerate(('embedding','confidence','router')):
  ph=m.init(jax.random.key(8+i),tokens,method=getattr(m,'forward_'+name))['params']
  p={**p,name+'_head':ph[name+'_head']}
 p=r._mutate_params(p)
 child=a.width_config(cfg,8);cp=a.width_slice(p,cfg,8);cm=a.SimpleAttentionNetwork(child)
 outputs={}
 for quant in (False,True):
  applied=a._quantize.cq_ste_params(cp,4) if quant else cp
  outputs['cq' if quant else 'fp32']={'logits':r._tensor_payload(cm.apply({'params':applied},tokens,quant=quant)[0]),'heads':{name:r._tensor_payload(cm.apply({'params':applied},tokens,method=getattr(cm,'forward_'+name),quant=quant)) for name in ('embedding','confidence','router')}}
 out={'upstream_pin':pin,'objectives':'confidence BCE, router CE, embedding normalized-output MSE; frozen hidden cells per upstream','heads':heads,'width':{'config':r._config_payload(cfg),'tensors':r._named_tensor_map(p),'tokens':[2,7,4,9,3],'child_config':r._config_payload(child),'child_tensors':r._named_tensor_map(cp),'outputs':outputs}}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n');print('wrote',args.output,args.output.stat().st_size)

if __name__=='__main__':main()
