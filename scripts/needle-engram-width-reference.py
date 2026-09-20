#!/usr/bin/env python3
"""CPU-only engram-bearing width parity from the pinned Needle 3 source.

Natural engram geometry and stable Hadamard blocks use width 1024 -> 512. Gzip keeps
full parent/child tensor evidence bounded on disk; Go needs only stdlib gzip.
"""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('ref', Path(__file__).with_name('needle-reference.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    pin = r._assert_upstream_pin(args.upstream)
    a = r._load_architecture(args.upstream)
    if jax.default_backend() != 'cpu':
        raise SystemExit('CPU-only reference')
    a._quantize.configure_deploy(act_bits=8, kv_bits=8)
    cfg = a.TransformerConfig(
        vocab_size=16, d_model=1024, num_heads=4, num_kv_heads=1,
        qk_head_dim=4, v_head_dim=4, num_layers=2, max_seq_len=16,
        mhc_lanes=2, qkv_conv_taps=3, engram_layers=(0,),
        engram_orders=(2, 3), engram_slots=17,
        global_layers=(1,), sliding_window=3, ladder_widths=(512,),
        dtype='float32', flash=False, remat=False,
        embedding_dim=4, embedding_probes=2, embedding_queries=2,
        confidence_probes=2, confidence_queries=2, router_probes=2, router_queries=2)
    m = a.SimpleAttentionNetwork(cfg)
    tokens = jnp.array([[2, 7, 4, 9, 3]], jnp.int32)
    p = m.init(jax.random.key(37), tokens)['params']
    for i, name in enumerate(('embedding', 'confidence', 'router')):
        ph = m.init(jax.random.key(38+i), tokens, method=getattr(m, 'forward_'+name))['params']
        p = {**p, name+'_head': ph[name+'_head']}
    p = r._mutate_params(p)
    # Deterministic, nonzero, head-distinguishing matrices avoid committing a
    # large random-weight fixture. Prime-period patterns compress well while
    # retained/dropped heads and grouped head-projection rows remain distinct.
    flat = flatten_dict(p)
    for key, value in list(flat.items()):
        if value.ndim >= 2:
            phase = sum('/'.join(key).encode()) % 251
            values = ((np.arange(value.size, dtype=np.int64) + phase) % 251 - 125)
            flat[key] = jnp.asarray((values.astype(np.float32) / 8192).reshape(value.shape))
    p = unflatten_dict(flat)
    child = a.width_config(cfg, 512)
    cp = a.width_slice(p, cfg, 512)
    cm = a.SimpleAttentionNetwork(child)
    outputs = {}
    for quant in (False, True):
        applied = a._quantize.cq_ste_params(cp, 4) if quant else cp
        outputs['cq' if quant else 'fp32'] = {
            'logits': r._tensor_payload(cm.apply({'params': applied}, tokens, quant=quant)[0]),
            'heads': {name: r._tensor_payload(cm.apply({'params': applied}, tokens,
                method=getattr(cm, 'forward_'+name), quant=quant))
                for name in ('embedding', 'confidence', 'router')},
        }
    out = {
        'upstream_pin': pin, 'config': r._config_payload(cfg),
        'tensors': r._named_tensor_map(p), 'tokens': [2, 7, 4, 9, 3],
        'child_config': r._config_payload(child),
        'child_tensors': r._named_tensor_map(cp),
        'child_cq_tensors': r._named_tensor_map(a._quantize.cq_ste_params(cp, 4)),
        'outputs': outputs,
    }
    raw = (json.dumps(out, separators=(',', ':'))+'\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    print('wrote', args.output, 'raw', len(raw), 'compressed', args.output.stat().st_size)


if __name__ == '__main__':
    main()
