#!/usr/bin/env python3
"""Generate flow_head_backward_pytorch.json from the pinned upstream module.

Run from kyutai-labs/pocket-tts@0acce6b2f390150267557770d2098c5caa9a18ac:

    .venv/bin/python /path/to/go-pherence/model/pockettts/testdata/\
generate_flow_head_backward_pytorch.py > /path/to/go-pherence/model/pockettts/\
testdata/flow_head_backward_pytorch.json
"""

import json

import torch
from torch import nn

from pocket_tts.modules.mlp import SimpleMLPAdaLN, TimestepEmbedder

UPSTREAM_REVISION = "0acce6b2f390150267557770d2098c5caa9a18ac"


def fill_linear(linear: nn.Linear, start: float) -> None:
    with torch.no_grad():
        flat = linear.weight.flatten()
        for i in range(flat.numel()):
            flat[i] = start + ((i * 7) % 13 - 6) * 0.025
        for i in range(linear.bias.numel()):
            linear.bias[i] = start / 3 + (i - 1) * 0.02


model = SimpleMLPAdaLN(
    in_channels=2,
    model_channels=4,
    out_channels=2,
    cond_channels=3,
    num_res_blocks=2,
    num_time_conds=2,
)
# The production topology uses 256 Fourier channels. Four channels retain the
# exact module arithmetic while keeping the all-parameter fixture reviewable.
model.time_embed = nn.ModuleList(
    [TimestepEmbedder(4, frequency_embedding_size=4, max_period=100) for _ in range(2)]
)
fill_linear(model.input_proj, 0.03)
fill_linear(model.cond_embed, -0.02)
fill_linear(model.time_embed[0].mlp[0], 0.01)
fill_linear(model.time_embed[0].mlp[2], -0.03)
fill_linear(model.time_embed[1].mlp[0], -0.015)
fill_linear(model.time_embed[1].mlp[2], 0.025)
with torch.no_grad():
    model.time_embed[0].mlp[3].alpha.copy_(torch.tensor([0.8, 1.1, 0.9, 1.2]))
    model.time_embed[1].mlp[3].alpha.copy_(torch.tensor([1.05, 0.95, 1.15, 0.85]))

block_specs = [
    (0.02, -0.01, 0.005, [1.1, 0.9, 1.2, 0.8], [0.02, -0.03, 0.01, 0.04]),
    (-0.025, 0.015, -0.004, [0.95, 1.05, 0.85, 1.15], [-0.01, 0.02, -0.04, 0.03]),
]
for block, (fc1, fc2, modulation, norm_weight, norm_bias) in zip(
    model.res_blocks, block_specs
):
    fill_linear(block.mlp[0], fc1)
    fill_linear(block.mlp[2], fc2)
    fill_linear(block.adaLN_modulation[1], modulation)
    with torch.no_grad():
        block.in_ln.weight.copy_(torch.tensor(norm_weight))
        block.in_ln.bias.copy_(torch.tensor(norm_bias))
fill_linear(model.final_layer.linear, 0.02)
fill_linear(model.final_layer.adaLN_modulation[1], -0.006)

condition = torch.tensor([0.2, -0.3, 0.5], dtype=torch.float32, requires_grad=True)
times = torch.tensor([0.25, 0.8], dtype=torch.float32, requires_grad=True)
latent = torch.tensor([-0.4, 0.7], dtype=torch.float32, requires_grad=True)
d_output = torch.tensor([0.6, -0.9], dtype=torch.float32)
output = model(condition, times[0:1], times[1:2], latent)
loss = (output * d_output).sum()
loss.backward()

fixture = {
    "schema": 1,
    "generator": f"pocket-tts .venv/bin/python (torch {torch.__version__})",
    "upstream_revision": UPSTREAM_REVISION,
    "dtype": "torch.float32",
    "condition": condition.detach().tolist(),
    "times": times.detach().tolist(),
    "input": latent.detach().tolist(),
    "d_output": d_output.tolist(),
    "output": output.detach().tolist(),
    "d_condition": condition.grad.tolist(),
    "d_times": times.grad.tolist(),
    "d_input": latent.grad.tolist(),
    "parameters": {
        name: value.detach().flatten().tolist()
        for name, value in model.state_dict().items()
    },
    "gradients": {
        name: value.grad.detach().flatten().tolist()
        for name, value in model.named_parameters()
    },
}
print(json.dumps(fixture, indent=2))
