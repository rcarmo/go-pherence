#!/usr/bin/env python3
"""Compare the native CLI block probe with one real PyTorch Qwen3 layer.
Needs a local OmniVoice checkpoint and an existing torch/transformers environment.
Does not load the whole model, download assets or run waveform synthesis.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path
import torch
from safetensors import safe_open
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', required=True, type=Path)
p.add_argument('--layer', type=int, default=0)
p.add_argument('--tokens', type=int, default=3)
a = p.parse_args()
if not 1 <= a.tokens <= 256:
    p.error('tokens must be 1..256')
torch.set_num_threads(2)
config = Qwen3Config(**json.loads((a.model/'config.json').read_text())['llm_config'])
config._attn_implementation = 'eager'
block = Qwen3DecoderLayer(config, layer_idx=a.layer).eval()
prefix = f'llm.layers.{a.layer}.'
with safe_open(a.model/'model.safetensors', framework='pt', device='cpu') as weights:
    state = {name[len(prefix):]: weights.get_tensor(name).float() for name in weights.keys() if name.startswith(prefix)}
block.load_state_dict(state, strict=True)
x = torch.tensor([math.sin(i*0.01)*0.1 for i in range(a.tokens*config.hidden_size)], dtype=torch.float32).reshape(1,a.tokens,config.hidden_size)
positions = torch.arange(a.tokens).unsqueeze(0)
with torch.inference_mode():
    rotary = Qwen3RotaryEmbedding(config)(x,positions)
    expected = block(x, attention_mask=None, position_ids=positions, position_embeddings=rotary).flatten()
raw = subprocess.check_output(['go','run','./cmd/audio/omnivoice','-mode','block','-model',str(a.model),'-layer',str(a.layer),'-tokens',str(a.tokens)], text=True)
actual = json.loads(raw)
error = max(abs(x-y) for x,y in zip(actual['first_values'],expected[:8].tolist()))
sum_error = abs(actual['sum']-expected.double().sum().item())
peak_error = abs(actual['peak']-expected.abs().max().item())
print(json.dumps({'first_8_max_error':error,'sum_error':sum_error,'peak_error':peak_error,'native':actual},indent=2))
# Reduction order differs between SIMD and PyTorch; these are float32 checks,
# not bit-exact claims. Tiny fixture tests compare every output element.
assert error < 1e-4 and peak_error < 1e-4 and sum_error < 1e-2
