#!/usr/bin/env python3
"""Generate flow_head_backward_pytorch.json from the pinned upstream module.

Run from kyutai-labs/pocket-tts@0acce6b2f390150267557770d2098c5caa9a18ac:

    .venv/bin/python /path/to/go-pherence/model/pockettts/testdata/\
generate_flow_head_backward_pytorch.py > /path/to/go-pherence/model/pockettts/\
testdata/flow_head_backward_pytorch.json
"""

import inspect
import json
import subprocess
from pathlib import Path

import torch
from torch import nn

from pocket_tts.modules.mlp import SimpleMLPAdaLN, TimestepEmbedder
from training.modules.samplers import LSD
from training.modules.utils import f_grad_x_only

UPSTREAM_REVISION = "0acce6b2f390150267557770d2098c5caa9a18ac"

checkout = Path.cwd().resolve()
actual_revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
).strip()
if actual_revision != UPSTREAM_REVISION:
    raise SystemExit(f"upstream revision {actual_revision}, want {UPSTREAM_REVISION}")
imports = {
    Path(inspect.getfile(SimpleMLPAdaLN)).resolve(): checkout / "pocket_tts" / "modules" / "mlp.py",
    Path(inspect.getfile(LSD)).resolve(): checkout / "training" / "modules" / "samplers.py",
    Path(inspect.getfile(f_grad_x_only)).resolve(): checkout / "training" / "modules" / "utils.py",
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
        "pocket_tts/modules/mlp.py",
        "training/modules/samplers.py",
        "training/modules/utils.py",
    ],
    cwd=checkout,
    check=True,
)


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

def time_function(all_times: torch.Tensor) -> torch.Tensor:
    return model(condition, all_times[0:1], all_times[1:2], latent)

jvp_times = []
for time_index in range(2):
    direction = torch.zeros_like(times)
    direction[time_index] = 1
    _, tangent = torch.func.jvp(time_function, (times,), (direction,))
    jvp_times.append(tangent.detach().tolist())

loss = (output * d_output).sum()
loss.backward()
ordinary_gradients = {
    name: value.grad.detach().flatten().tolist()
    for name, value in model.named_parameters()
}
ordinary_inputs = {
    "d_condition": condition.grad.detach().tolist(),
    "d_times": times.grad.detach().tolist(),
    "d_input": latent.grad.detach().tolist(),
}

d_tangent = torch.tensor([-0.35, 0.45], dtype=torch.float32)
mixed = []
for time_index in range(2):
    model.zero_grad(set_to_none=True)
    condition.grad = None
    times.grad = None
    latent.grad = None
    direction = torch.zeros_like(times)
    direction[time_index] = 1
    output_mixed, tangent_mixed = torch.func.jvp(
        time_function, (times,), (direction,)
    )
    mixed_loss = (output_mixed * d_output).sum() + (
        tangent_mixed * d_tangent
    ).sum()
    mixed_loss.backward()
    mixed.append(
        {
            "time_index": time_index,
            "d_output": d_output.tolist(),
            "d_tangent": d_tangent.tolist(),
            "loss": mixed_loss.item(),
            "d_condition": condition.grad.detach().tolist(),
            "d_times": times.grad.detach().tolist(),
            "d_input": latent.grad.detach().tolist(),
            "gradients": {
                name: value.grad.detach().flatten().tolist()
                for name, value in model.named_parameters()
            },
        }
    )

model.zero_grad(set_to_none=True)
condition.grad = None
s = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)
t = torch.tensor([0.8], dtype=torch.float32, requires_grad=True)
noise = torch.tensor([-0.4, 0.7], dtype=torch.float32, requires_grad=True)
target = torch.tensor([0.6, -0.2], dtype=torch.float32, requires_grad=True)
logvar = torch.tensor(0.12, dtype=torch.float32, requires_grad=True)
x_s = s * target + (1 - s) * noise


def primary(s_value: torch.Tensor, t_value: torch.Tensor, x_value: torch.Tensor) -> torch.Tensor:
    return model(condition, s_value, t_value, x_value)


velocity, time_derivative = torch.func.jvp(
    primary,
    (s, t, x_s),
    (torch.zeros_like(s), torch.ones_like(t), torch.zeros_like(x_s)),
)
x_t = x_s + (t - s) * velocity
dxdt = velocity + (t - s) * time_derivative
endpoint = f_grad_x_only(lambda y: model(condition, t, t, y), x_t)
residual = dxdt - endpoint
raw_square = residual.square().sum()
lsd_loss = raw_square * logvar.exp() / noise.shape[-1] - logvar
lsd_loss.backward()
lsd = {
    "s": s.detach().item(),
    "t": t.detach().item(),
    "condition": condition.detach().tolist(),
    "noise": noise.detach().tolist(),
    "target": target.detach().tolist(),
    "log_variance": logvar.detach().item(),
    "velocity": velocity.detach().tolist(),
    "time_derivative": time_derivative.detach().tolist(),
    "endpoint": endpoint.detach().tolist(),
    "residual": residual.detach().tolist(),
    "raw_square": raw_square.detach().item(),
    "loss": lsd_loss.detach().item(),
    "d_condition": condition.grad.detach().tolist(),
    "d_s": s.grad.detach().item(),
    "d_t": t.grad.detach().item(),
    "d_noise": noise.grad.detach().tolist(),
    "d_target": target.grad.detach().tolist(),
    "d_log_variance": logvar.grad.detach().item(),
    "gradients": {
        name: value.grad.detach().flatten().tolist()
        for name, value in model.named_parameters()
    },
}

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
    "d_condition": ordinary_inputs["d_condition"],
    "d_times": ordinary_inputs["d_times"],
    "jvp_times": jvp_times,
    "d_input": ordinary_inputs["d_input"],
    "parameters": {
        name: value.detach().flatten().tolist()
        for name, value in model.state_dict().items()
    },
    "gradients": ordinary_gradients,
    "mixed": mixed,
    "lsd_minimal": lsd,
}
print(json.dumps(fixture, indent=2))
