#!/usr/bin/env python3
"""Pinned CPU Needle2 archive writer/reader and reconstructed inference fixture."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

spec = importlib.util.spec_from_file_location('ref', Path(__file__).with_name('needle-reference.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

PIN = '741ee892c5f8c4f5c0bb467c9566ea7a1eba919b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    args = parser.parse_args()
    pkg = 'needle2archive'
    sys.modules[pkg] = types.ModuleType(pkg)
    sys.modules[pkg].__path__ = []
    for name in ('quantize', 'architecture', 'export'):
        src = subprocess.check_output(['git','-C',str(args.upstream),'show',f'{PIN}:needle/model/{name}.py'],text=True)
        mod = types.ModuleType(pkg+'.'+name);mod.__package__=pkg
        sys.modules[mod.__name__]=mod
        exec(compile(src,f'{PIN}/{name}.py','exec'),mod.__dict__)
    a,ex = sys.modules[pkg+'.architecture'],sys.modules[pkg+'.export']
    if jax.default_backend()!='cpu': raise SystemExit('CPU-only reference')
    a._quantize.configure_deploy(act_bits=8,kv_bits=8)
    f=json.loads((Path(__file__).parent.parent/'model/needle/testdata/needle2-heads.json').read_text())
    config={**f['config'],'vocab_size':280,'out_vocab':0,'kv_bits':8}
    cfg=a.TransformerConfig(**config)
    flat={tuple(k.split('/')):np.array(v['data'],np.float32).reshape(v['shape']) for k,v in f['tensors'].items()}
    flat[('embedding','embedding')]=np.resize(flat[('embedding','embedding')],(280,8))
    params=unflatten_dict(flat)
    # Reuse the synthetic SentencePiece surface, but the pinned v2 serializer.
    tokmod=types.ModuleType(pkg+'.tokenizer')
    tokmod.CHAT_MARKERS=['<|im_start|>','<|im_end|>','<think>','</think>','<tools>','</tools>','<tool_call>','</tool_call>','<tool_result>','</tool_result>']
    tokmod.PAD_ID,tokmod.EOS_ID,tokmod.BOS_ID,tokmod.UNK_ID=0,1,2,3
    sys.modules[tokmod.__name__]=tokmod
    class SP:
        pieces=['<pad>','<eos>','<bos>','<unk>']+[f'<0x{i:02X}>' for i in range(256)]+['▁','h','e','l','o','▁h','he','ll','hello','▁hello']+tokmod.CHAT_MARKERS
        def GetPieceSize(self):return len(self.pieces)
        def IdToPiece(self,i):return self.pieces[i]
        def GetScore(self,i):return float(i%11)/8
        def IsControl(self,i):return i<3
        def IsUnknown(self,i):return i==3
        def IsByte(self,i):return 4<=i<260
    tok=types.SimpleNamespace(sp=SP(),vocab_size=280)
    args.outdir.mkdir(parents=True,exist_ok=True)
    path=args.outdir/'needle2.cact'
    path.write_bytes(ex.build_export(params,cfg,tokenizer=tok,kv_window=16))
    header,records=ex.read_export(path)
    header['codebook']=header['codebook'].tolist()
    named=ex._tensors(params,cfg,4,128)
    restored={}
    for record,data in zip(named,records):
        name=record.name
        if name=='embedding':key=('embedding','embedding');restored[key]=data;continue
        if name.startswith('layer'):
            layer=int(name[5:7]);suffix=name.split('.')[1]
            names={'norm_in':('ZCRMSNorm_0','scale'),'post_norm':('post_attn_norm','scale'),'pre_hada':('pre_hada_norm','scale'),'attn_gate':('attn_gate',)}
            if suffix in names:tail=names[suffix]
            elif suffix in ('d1','d2','d3'):tail=('hadamard_mlp',suffix)
            else:tail=('self_attn',suffix,'scale' if suffix.endswith('norm') else 'kernel')
            key=('stack','layers','block')+tail
            if key not in restored:restored[key]=np.zeros_like(flat[key])
            restored[key][layer]=data.T if tail[-1]=='kernel' else data.reshape(restored[key][layer].shape)
        elif name.startswith('mhc_'):
            key=('stack',name)
            if name.startswith('mhc_phi'):
                L,nC,lanes=flat[key].shape;data=data.reshape(L,lanes,nC).transpose(0,2,1)
            restored[key]=data
        elif name.startswith('engram'):
            prefix,suffix=name.split('.');key=(f'engrams_{int(prefix[6:])}',)
            if suffix=='tables':key+=('embedding',);data=data.reshape(flat[key].shape)
            elif suffix=='taps':key+=('taps',)
            else:key+=(suffix,'kernel');data=data.T
            restored[key]=data
        elif name=='final_norm':restored[('stack','final_norm','scale')]=data
        elif name=='heads.manifest':continue
        else:
            head,suffix=name.split('.')
            if head=='contrastive_head' and suffix=='bias':continue
            key=(head,'probes') if suffix=='probes' else (head,'proj','kernel' if suffix=='proj' else 'bias')
            restored[key]=data.T if suffix=='proj' else data
    # Upstream head module requests temperature but archive omits it. The fixed
    # value only allows reference application; exported vectors do not use it.
    restored[('contrastive_head','log_temp')]=flat[('contrastive_head','log_temp')]
    p=unflatten_dict({k:jnp.array(v) for k,v in restored.items()})
    m=a.SimpleAttentionNetwork(cfg);ids=jnp.array([f['tokens']],jnp.int32)
    logits=m.apply({'params':p},ids,quant=True)
    heads={}
    for name,method in [('contrastive',m.encode_contrastive),('confidence',m.forward_confidence)]:
        heads[name]=r._tensor_payload(m.apply({'params':p},ids,method=method,quant=True))
    sidecar={k:config[k] for k in ['generation','vocab_size','d_model','attn_dim','num_heads','num_kv_heads','num_layers','max_seq_len','mhc_lanes','engram_orders','engram_heads','engram_slots','engram_layers','rope_theta','pad_token_id','contrastive_dim']}
    sidecar['dtype']='float32'
    (args.outdir/'needle2-archive-config.json').write_text(json.dumps(sidecar,indent=2)+'\n')
    out={'upstream_pin':PIN,'header':header,'tokens':f['tokens'],'logits':r._tensor_payload(logits[0]),'heads':heads,'records':[{'name':t.name,'shape':list(t.shape),'data':np.asarray(v).reshape(-1).tolist()} for t,v in zip(named,records)]}
    (args.outdir/'needle2-archive-reference.json').write_text(json.dumps(out,indent=2)+'\n')
    print('wrote',path,path.stat().st_size,len(records),'records')


if __name__=='__main__':main()
