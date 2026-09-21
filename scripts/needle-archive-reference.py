#!/usr/bin/env python3
"""Generate tiny Needle3 archive/byte-tokenizer fixtures with the pinned exporter.
Offline CPU-only; no native engine or production weights are executed.
"""
import argparse
import importlib.util
import json
import struct
from pathlib import Path

spec=importlib.util.spec_from_file_location('reference',Path(__file__).with_name('needle-reference.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import unflatten_dict


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
 pin=r._assert_upstream_pin(args.upstream);arch=r._load_architecture(args.upstream)
 export=r._load_module(r.PACKAGE_NAME+'.model.export',args.upstream/'needle/model/export.py')
 # Synthetic export tokenizer: known merges, Unicode and overlapping markers,
 # plus all 256 byte fallback pieces, independent of production token IDs.
 pieces=['<pad>','</s>','<s>','<unk>']+[f'<0x{v:02X}>' for v in range(256)]
 types=[2,2,2,1]+[4]*256;scores=[0.]*260
 for i,s in enumerate(['▁','a','b','c','h','e','l','o','ab','bc','he','ll','hello','▁hello','▁a','é','<|im_start|>','<|im_end|>','<tag>','<tag>long']):
  pieces.append(s);types.append(3 if s.startswith('<') else 0);scores.append(float(i%5))
 blob=bytearray(struct.pack('<IIIIIBBH',len(pieces),0,1,2,3,1,1,0))
 for s,kind,score in zip(pieces,types,scores):
  b=s.encode();blob.extend(struct.pack('<fBH',score,kind,len(b)));blob.extend(b)
 tok=export.RefTokenizer(export.parse_tokenizer_blob(blob))
 texts=['','hello','hello  hello','abc abbc','é 🚀 中文','\n\t','<|im_start|>user\nhello<|im_end|>','<tag>long<tag>','  leading  trailing ','a\u00a0b']
 cases=[{'text':s,'ids':tok.encode(s),'decoded':tok.decode(tok.encode(s))} for s in texts]
 decodeids=[[4+0xff,4+0xff],[4+0xe2,4+0x82],[4+0xe2,4+0x82,4+0x41],[4+0xe0,4+0x80,4+0x80],[4+0xed,4+0xa0,4+0x80]]
 decode=[{'ids':ids,'text':tok.decode(ids)} for ids in decodeids]
 f=json.loads((Path(__file__).parent.parent/'model/needle/testdata/needle3-extended.json').read_text())['base']
 config=dict(f['config']);config['vocab_size']=len(pieces);config['out_vocab']=16
 params=unflatten_dict({tuple(k.split('/')):np.array(v['data'],np.float32).reshape(v['shape']) for k,v in f['tensors'].items()})
 params['embedding']['embedding']=np.tile(params['embedding']['embedding'],(18,1))[:len(pieces)]
 cfg=arch.TransformerConfig(**config)
 args.output_dir.mkdir(parents=True,exist_ok=True)
 path=args.output_dir/'needle3.cact';export.write_export(params,cfg,str(path),tokenizer=blob,kv_window=16)
 geometry,decoded=export.read_export(path)
 refs=[{'shape':list(v.shape),'data':np.asarray(v,np.float32).reshape(-1).tolist()} if not isinstance(v,(bytes,bytearray)) else {'raw_hex':bytes(v).hex()} for v in decoded]
 # Independently map positional decoded exporter tensors back into the JAX
 # parameter tree; compare model execution, not just byte decoding.
 from flax.traverse_util import flatten_dict
 restored={k:np.array(v,copy=True) for k,v in flatten_dict(params).items()}
 block={ 'norm_in':'ZCRMSNorm_0/scale','q_proj':'self_attn/q_proj/kernel','k_proj':'self_attn/k_proj/kernel','v_proj':'self_attn/v_proj/kernel','gate_proj':'self_attn/gate_proj/kernel','out_proj':'self_attn/out_proj/kernel','q_norm':'self_attn/q_norm/scale','k_norm':'self_attn/k_norm/scale','post_norm':'post_attn_norm/scale','attn_gate':'attn_gate','pre_hada':'pre_hada_norm/scale'}
 for spec,arr in zip(export._tensors(params,cfg,4,128),decoded):
  name=spec.name
  if name=='embedding':restored[('embedding','embedding')]=arr
  elif name.startswith('layer'):
   layer=int(name[5:7]);key=name.split('.',1)[1]
   sub=block.get(key,('self_attn/' if key.endswith('_taps') else 'hadamard_mlp/')+key)
   target=('stack','layers','block',*sub.split('/'))
   val=arr.T if sub.endswith('/kernel') else arr
   restored[target][layer]=val.reshape(restored[target][layer].shape)
  elif name.startswith('mhc_phi'):
   target=('stack',name);shape=restored[target].shape
   restored[target]=arr.reshape(shape[0],shape[2],shape[1]).transpose(0,2,1)
  elif name.startswith('mhc_'):restored[('stack',name)]=arr
  elif name.startswith('engram'):
   site,key=name.split('.',1);site='engrams_'+site[6:]
   sub={'tables':('embedding',),'key_proj':('key_proj','kernel'),'value_proj':('value_proj','kernel'),'taps':('taps',)}[key]
   target=(site,*sub);val=arr.T if key.endswith('_proj') else arr
   restored[target]=val.reshape(restored[target].shape)
  elif name=='final_norm':restored[('stack','final_norm','scale')]=arr
  elif '_head.' in name:
   head,key=name.split('.',1)
   target=(head,'proj','kernel') if key=='proj' else (head,'proj','bias') if key=='bias' else (head,key)
   if target in restored:
    val=arr.T if key=='proj' else arr
    restored[target]=val.reshape(restored[target].shape)
 # head_weight is an export-time QAT hook. Decoded head tensors are already CQ;
 # bypass only that hook, retaining upstream forward/masking/pooling math.
 arch.head_weight=lambda w,quant,**kwargs:w
 arch._quantize.configure_deploy(act_bits=8,kv_bits=8)
 model=arch.SimpleAttentionNetwork(cfg);weights=unflatten_dict({k:jnp.asarray(v) for k,v in restored.items()});ids=[2,7,4,9,3];batch=jnp.asarray([ids])
 logits=model.apply({'params':weights},batch,quant=True)
 heads={}
 for name,method in [('embedding',model.forward_embedding),('confidence',model.forward_confidence),('router',model.forward_router)]:
  heads[name]=np.asarray(model.apply({'params':weights},batch,method=method,quant=True)).reshape(-1).tolist()
 out={'pin':pin,'records':refs,'tokenizer':{'pieces':len(pieces),'encode':cases,'decode':decode},'model':{'tokens':ids,'logits':r._tensor_payload(logits[0]),'heads':heads,'note':'Pinned JAX architecture evaluated on decoded exported tensors, not a native runtime comparison'}}
 (args.output_dir/'archive-reference.json').write_text(json.dumps(out,indent=2)+'\n')
 (args.output_dir/'tokenizer.bin').write_bytes(blob)
 print(f'wrote {path}: {len(decoded)} records; {len(pieces)} pieces; {len(cases)} encode and {len(decode)} invalid-byte decode cases')

if __name__=='__main__':main()
