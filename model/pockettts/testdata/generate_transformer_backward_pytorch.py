#!/usr/bin/env python3
"""Generate transformer_backward_pytorch.json from the pinned upstream module.

Run from kyutai-labs/pocket-tts@0acce6b2f390150267557770d2098c5caa9a18ac.
"""

import inspect
import json
import subprocess
from pathlib import Path

import torch
from torch import nn

from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.transformer import StreamingTransformer

UPSTREAM_REVISION = "0acce6b2f390150267557770d2098c5caa9a18ac"
checkout = Path.cwd().resolve()
actual_revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
).strip()
if actual_revision != UPSTREAM_REVISION:
    raise SystemExit(f"upstream revision {actual_revision}, want {UPSTREAM_REVISION}")
imports = {
    Path(inspect.getfile(StreamingTransformer)).resolve(): checkout
    / "pocket_tts/modules/transformer.py",
    Path(inspect.getfile(StreamingMultiheadAttention)).resolve(): checkout
    / "pocket_tts/modules/attention.py",
    Path(inspect.getfile(RotaryEmbedding)).resolve(): checkout
    / "pocket_tts/modules/rope.py",
}
for actual, expected in imports.items():
    if actual != expected:
        raise SystemExit(f"imported {actual}, want {expected}")
subprocess.run(
    [
        "git",
        "diff",
        "--quiet",
        "HEAD",
        "--",
        "pocket_tts/modules/transformer.py",
        "pocket_tts/modules/attention.py",
        "pocket_tts/modules/rope.py",
        "pocket_tts/modules/layer_scale.py",
    ],
    cwd=checkout,
    check=True,
)


def fill_linear(linear: nn.Linear, start: float) -> None:
    if linear.bias is not None:
        raise AssertionError("fixture transformer linears must be bias-free")
    with torch.no_grad():
        flat = linear.weight.flatten()
        for i in range(flat.numel()):
            flat[i] = start + ((i * 7) % 13 - 6) * 0.025


model = StreamingTransformer(
    d_model=4,
    num_heads=2,
    num_layers=1,
    layer_scale=1.0,
    dim_feedforward=6,
    context=2,
    max_period=100.0,
)
layer = model.layers[0]
fill_linear(layer.self_attn.in_proj, 0.01)
fill_linear(layer.self_attn.out_proj, -0.02)
fill_linear(layer.linear1, 0.03)
fill_linear(layer.linear2, -0.015)
with torch.no_grad():
    layer.norm1.weight.copy_(torch.tensor([1.1, 0.9, 1.2, 0.8]))
    layer.norm1.bias.copy_(torch.tensor([0.02, -0.03, 0.01, 0.04]))
    layer.norm2.weight.copy_(torch.tensor([0.95, 1.05, 0.85, 1.15]))
    layer.norm2.bias.copy_(torch.tensor([-0.01, 0.02, -0.04, 0.03]))
    layer.layer_scale_1.scale.copy_(torch.tensor([0.8, 1.1, 0.9, 1.2]))
    layer.layer_scale_2.scale.copy_(torch.tensor([1.05, 0.95, 1.15, 0.85]))

final = nn.LayerNorm(4, eps=1e-5)
with torch.no_grad():
    final.weight.copy_(torch.tensor([1.02, 0.98, 1.08, 0.92]))
    final.bias.copy_(torch.tensor([0.01, -0.02, 0.03, -0.04]))

sequence = torch.tensor(
    [[[0.2, -0.4, 0.1, 0.5], [-0.3, 0.7, 0.6, -0.2], [0.8, -0.1, 0.4, -0.5]]],
    dtype=torch.float32,
    requires_grad=True,
)
d_output = torch.tensor(
    [[[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8], [-0.3, 0.4, -0.5, 0.6]]],
    dtype=torch.float32,
)
output = final(model(sequence, model_state=None))
loss = (output * d_output).sum()
loss.backward()

parameters = {
    f"transformer.{name}": value.detach().flatten().tolist()
    for name, value in model.state_dict().items()
}
parameters.update(
    {f"final.{name}": value.detach().flatten().tolist() for name, value in final.state_dict().items()}
)
gradients = {
    f"transformer.{name}": value.grad.detach().flatten().tolist()
    for name, value in model.named_parameters()
}
gradients.update(
    {f"final.{name}": value.grad.detach().flatten().tolist() for name, value in final.named_parameters()}
)
fixture = {
    "schema": 1,
    "generator": f"pocket-tts .venv/bin/python (torch {torch.__version__})",
    "upstream_revision": UPSTREAM_REVISION,
    "dtype": "torch.float32",
    "rows": 3,
    "width": 4,
    "heads": 2,
    "context": 2,
    "max_period": 100.0,
    "sequence": sequence.detach().flatten().tolist(),
    "d_output": d_output.flatten().tolist(),
    "output": output.detach().flatten().tolist(),
    "d_sequence": sequence.grad.detach().flatten().tolist(),
    "parameters": parameters,
    "gradients": gradients,
}
print(json.dumps(fixture, indent=2))
