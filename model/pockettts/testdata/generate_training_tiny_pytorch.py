#!/usr/bin/env python3
"""Generate training_tiny_pytorch.json with an independent PyTorch oracle.

Run from the pinned pocket-tts checkout so its virtual environment supplies
PyTorch, and write into the Go checkout:

    .venv/bin/python /path/to/go-pherence/model/pockettts/testdata/\
generate_training_tiny_pytorch.py > /path/to/go-pherence/model/pockettts/\
testdata/training_tiny_pytorch.json
"""

import json

import torch

UPSTREAM_REVISION = "0acce6b2f390150267557770d2098c5caa9a18ac"

hidden = torch.tensor(
    [[0.2, -0.4], [0.1, 0.3], [-0.2, 0.5], [0.7, -0.1]], dtype=torch.float32
)
noise = torch.tensor(
    [[0.3, -0.2], [0.4, 0.1], [-0.5, 0.2], [0.6, -0.3]], dtype=torch.float32
)
target = torch.tensor(
    [[0.8, 0.1], [-0.1, 0.7], [0.2, -0.4], [0.5, 0.9]], dtype=torch.float32
)
times = torch.tensor([0.2, 0.7, 0.4, 0.9], dtype=torch.float32)
mask = torch.tensor([True, True, False, False])

flow_weight = torch.tensor(
    [[0.1, -0.2, 0.3, -0.4, 0.5], [-0.3, 0.2, 0.4, 0.1, -0.2]],
    dtype=torch.float32,
    requires_grad=True,
)
flow_bias = torch.tensor([0.05, -0.07], dtype=torch.float32, requires_grad=True)
flow_logvar = torch.tensor(0.15, dtype=torch.float32, requires_grad=True)
eos_weight = torch.tensor([0.25, -0.35], dtype=torch.float32, requires_grad=True)
eos_bias = torch.tensor(0.08, dtype=torch.float32, requires_grad=True)

x_t = times[:, None] * target + (1 - times[:, None]) * noise
flow_input = torch.cat([hidden, x_t, times[:, None]], dim=1)
prediction = flow_input[mask] @ flow_weight.T + flow_bias
velocity = (target - noise)[mask]
flow_square = (prediction - velocity).square().sum(-1)
flow_loss = (
    flow_square * flow_logvar.exp() / target.shape[-1] - flow_logvar
).mean()

eos_logits = hidden @ eos_weight + eos_bias
is_eos = ~mask
is_eos[0] = False
shifted_mask = torch.cat([mask[:1], mask[:-1]])
eos_per_row = is_eos * torch.nn.functional.softplus(-eos_logits) + (
    ~is_eos
) * torch.nn.functional.softplus(eos_logits)
eos_loss = (eos_per_row * shifted_mask).sum() / shifted_mask.sum().clamp(min=1)
loss = 0.75 * flow_loss + 0.1 * eos_loss
loss.backward()

parameters = {
    "flow.weight": flow_weight,
    "flow.bias": flow_bias,
    "flow.logvar": flow_logvar,
    "eos.weight": eos_weight,
    "eos.bias": eos_bias,
}
gradients = {
    name: value.grad.detach().flatten().tolist() for name, value in parameters.items()
}
before = {
    name: value.detach().flatten().tolist() for name, value in parameters.items()
}
optimizer = torch.optim.AdamW(
    list(parameters.values()),
    lr=0.001,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)
optimizer.step()
after = {
    name: value.detach().flatten().tolist() for name, value in parameters.items()
}
ema = {
    name: [0.9 * old + 0.1 * new for old, new in zip(before[name], after[name])]
    for name in before
}

fixture = {
    "schema": 1,
    "generator": f"pocket-tts .venv/bin/python (torch {torch.__version__})",
    "upstream_revision": UPSTREAM_REVISION,
    "dtype": "torch.float32",
    "hidden": hidden.flatten().tolist(),
    "noise": noise.flatten().tolist(),
    "target": target.flatten().tolist(),
    "times": times.tolist(),
    "mask": mask.tolist(),
    "parameters": before,
    "metrics": {
        "flow_diagonal": flow_loss.item(),
        "eos": eos_loss.item(),
        "loss": loss.item(),
    },
    "gradients": gradients,
    "adamw": {
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "parameters_after_step": after,
        "ema_decay": 0.9,
        "ema_after_step": ema,
    },
}
print(json.dumps(fixture, indent=2))
