#!/usr/bin/env python3
"""Offline Needle 2 FP32 or CQ-W4/A8 parity fixture from an explicitly pinned Git archive.
Uses the same CPU-only environment and serialization as needle-reference.py.
"""
import argparse
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location('needle_reference', Path(__file__).with_name('needle-reference.py'))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
PIN = '741ee892c5f8c4f5c0bb467c9566ea7a1eba919b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--quantized', action='store_true')
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.traverse_util import flatten_dict, unflatten_dict
    # Extract only known model Python files; do not execute a mutable v2 worktree.
    with tempfile.TemporaryDirectory(prefix='needle2-reference-') as tmp:
        root = Path(tmp)
        model_dir = root / 'needle' / 'model'
        model_dir.mkdir(parents=True)
        for name in ('architecture.py', 'quantize.py'):
            data = subprocess.run(['git', '-C', str(args.upstream), 'show', f'{PIN}:needle/model/{name}'], check=True, capture_output=True).stdout
            (model_dir / name).write_bytes(data)
        arch = ref._load_architecture(root)
        if jax.default_backend() != 'cpu':
            raise SystemExit('CPU-only reference')
        # Pinned v2 maps requested KV8 to KV_BITS=0 (full-precision KV).
        arch._quantize.configure_deploy(act_bits=8, kv_bits=8)
        config = arch.TransformerConfig(vocab_size=16, d_model=8, attn_dim=8, num_heads=2,
            num_kv_heads=1, num_layers=2, max_seq_len=16, dtype='float32', flash=False,
            engram_orders=(2, 3), engram_heads=1, engram_slots=8, engram_layers=(0,), mhc_lanes=2)
        model = arch.SimpleAttentionNetwork(config)
        tokens = jnp.asarray([ref.TOKENS], jnp.int32)
        params = ref._mutate_params(model.init(jax.random.key(ref.SEED), tokens, quant=False)['params'])
        # Upstream load_checkpoint strips MTP-only auxiliary training parameters.
        params = {key: value for key, value in params.items() if not key.startswith('mtp_')}
        def loss_fn(p):
            applied = arch._quantize.cq_ste_params(p, 4) if args.quantized else p
            logits = model.apply({'params': applied}, tokens, quant=args.quantized)
            log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
            loss = -jnp.take_along_axis(log_probs, tokens[:, 1:, None], axis=-1).mean()
            return loss, logits
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        conf = ref._config_payload(config)
        conf['generation'] = 2
        out = {'schema': 'go-pherence/needle2-cpu-parity/v1',
            'generator': {'upstream_pin': PIN, 'device': jax.default_backend(), 'dtype': 'float32', 'quant': args.quantized,
                          'kv_numerics': 'fp32 (pinned configure_deploy maps kv_bits>=8 to off)',
                          'jax': jax.__version__, 'flax': ref.flax.__version__, 'script': 'scripts/needle2-reference.py'},
            'config': conf, 'tokens': ref.TOKENS, 'targets': ref.TOKENS[1:],
            'tensors': ref._named_tensor_map(params), 'logits': ref._tensor_payload(logits[0]),
            'loss': float(loss), 'gradients': ref._named_tensor_map(grads),
            'permutations': {'p1': {'data': []}, 'p2': {'data': []}}}
        if args.quantized:
            out['cq_tensors'] = ref._named_tensor_map(arch._quantize.cq_ste_params(params, 4))
            out['a8_logits'] = ref._tensor_payload(model.apply({'params':params},tokens,quant=True)[0])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, sort_keys=True, indent=2) + '\n')
        print(f'Needle2 CPU: {len(out["tensors"])} tensors, loss {float(loss):.9f}; {args.output}')

if __name__ == '__main__':
    main()
