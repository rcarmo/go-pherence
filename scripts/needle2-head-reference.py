#!/usr/bin/env python3
"""Pinned CPU Needle2 contrastive/confidence pooling and frozen-trunk gradients."""
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
from flax.traverse_util import unflatten_dict

PIN = '741ee892c5f8c4f5c0bb467c9566ea7a1eba919b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.modules['needle2head'] = types.ModuleType('needle2head')
    sys.modules['needle2head'].__path__ = []
    for name in ('quantize', 'architecture'):
        source = subprocess.check_output(['git', '-C', str(args.upstream), 'show', f'{PIN}:needle/model/{name}.py'], text=True)
        mod = types.ModuleType('needle2head.'+name)
        mod.__package__ = 'needle2head'
        sys.modules[mod.__name__] = mod
        exec(compile(source, f'{PIN}/{name}.py', 'exec'), mod.__dict__)
    a = sys.modules['needle2head.architecture']
    if jax.default_backend() != 'cpu':
        raise SystemExit('CPU-only reference')
    base = json.loads((Path(__file__).parent.parent/'model/needle/testdata/needle2.json').read_text())
    config = {**base['config'], 'contrastive_dim': 4}
    model = a.SimpleAttentionNetwork(a.TransformerConfig(**config))
    ids = jnp.array([base['tokens']], jnp.int32)
    params = unflatten_dict({tuple(k.split('/')): jnp.array(v['data'], jnp.float32).reshape(v['shape']) for k, v in base['tensors'].items()})
    for seed, kind in enumerate(('contrastive', 'confidence')):
        head = model.init(jax.random.key(61+seed), ids, method=getattr(model, '_encode_contrastive' if kind == 'contrastive' else 'forward_confidence'))['params']
        params = {**params, kind+'_head': head[kind+'_head']}
    heads = {}
    for kind, target in [('contrastive', [1., 0., 0., 0.]), ('confidence', [1.])]:
        method = getattr(model, '_encode_contrastive' if kind == 'contrastive' else 'forward_confidence')
        def loss_fn(p):
            output = model.apply({'params': p}, ids, method=method, quant=False)
            if kind == 'contrastive':
                output, _ = output
                loss = jnp.mean((output-jnp.array(target, jnp.float32))**2)
            else:
                loss = jnp.mean(jax.nn.softplus(output)-output*target[0])
            return loss, output
        (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        padded = model.apply({'params': params}, jnp.array([[2, 0, 4, 0, 3]], jnp.int32), method=method, quant=False)
        if kind == 'contrastive':
            padded, _ = padded
        heads[kind] = {'target': target, 'loss': float(loss), 'output': r._tensor_payload(output), 'padded_output': r._tensor_payload(padded), 'gradients': r._named_tensor_map(grads)}
    out = {'upstream_pin': PIN, 'config': config, 'tokens': base['tokens'], 'padded_tokens': [2, 0, 4, 0, 3], 'tensors': r._named_tensor_map(params), 'heads': heads, 'objective': 'Frozen-trunk BCE and normalized embedding MSE; not upstream contrastive pair loss or calibration'}
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print('wrote', args.output, args.output.stat().st_size)


if __name__ == '__main__':
    main()
