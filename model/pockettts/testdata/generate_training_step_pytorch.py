#!/usr/bin/env python3
"""Generate training_step_pytorch.json by calling pinned TrainableTTS.forward."""

import inspect
import json
import subprocess
from pathlib import Path

import torch
from torch import nn

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.mlp import SimpleMLPAdaLN, TimestepEmbedder
from pocket_tts.modules.transformer import StreamingTransformer
from training.args import TrainArgs
from training.modules.model import TrainableTTS
from training.modules.samplers import LSD
from training.modules.utils import f_grad_x_only

REVISION = "0acce6b2f390150267557770d2098c5caa9a18ac"
root = Path.cwd().resolve()
actual_revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True
).strip()
if actual_revision != REVISION:
    raise SystemExit(f"upstream revision {actual_revision}, want {REVISION}")
imports = {
    Path(inspect.getfile(FlowLMModel)).resolve(): root / "pocket_tts/models/flow_lm.py",
    Path(inspect.getfile(SimpleMLPAdaLN)).resolve(): root / "pocket_tts/modules/mlp.py",
    Path(inspect.getfile(StreamingTransformer)).resolve(): root / "pocket_tts/modules/transformer.py",
    Path(inspect.getfile(TrainableTTS)).resolve(): root / "training/modules/model.py",
    Path(inspect.getfile(LSD)).resolve(): root / "training/modules/samplers.py",
    Path(inspect.getfile(f_grad_x_only)).resolve(): root / "training/modules/utils.py",
}
for actual, expected in imports.items():
    if actual != expected:
        raise SystemExit(f"imported {actual}, want {expected}")
subprocess.run(
    [
        "git", "diff", "--quiet", "HEAD", "--",
        "pocket_tts/models/flow_lm.py",
        "pocket_tts/modules/mlp.py",
        "pocket_tts/modules/transformer.py",
        "pocket_tts/modules/attention.py",
        "pocket_tts/modules/rope.py",
        "pocket_tts/modules/layer_scale.py",
        "training/args.py",
        "training/modules/conditioner.py",
        "training/modules/model.py",
        "training/modules/samplers.py",
        "training/modules/utils.py",
    ],
    cwd=root,
    check=True,
)


def fill_linear(linear: nn.Linear, start: float) -> None:
    with torch.no_grad():
        for i in range(linear.weight.numel()):
            linear.weight.flatten()[i] = start + ((i * 7) % 13 - 6) * 0.025
        if linear.bias is not None:
            for i in range(linear.bias.numel()):
                linear.bias[i] = start / 3 + (i - 1) * 0.02


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(6, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


class DeterministicLSD(LSD):
    def __init__(self):
        super().__init__(p_equal=0.75, w_t_dims=3, w_t_depth=2, normalize=True, distill_prob=1.0)
        self.diagonal_times = torch.tensor([[0.2], [0.7]])
        self.start_times = torch.tensor([[0.15], [0.3]])
        self.end_times = torch.tensor([[0.8], [0.65]])
        self.recorded_noise = None

    def sample_t(self, x: torch.Tensor) -> torch.Tensor:
        return self.diagonal_times.to(x)

    def sample_s_t(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.start_times.to(x), self.end_times.to(x)

    def loss(self, v_t, x_0: torch.Tensor, x_1: torch.Tensor):
        self.recorded_noise = x_0.detach().clone()
        return super().loss(v_t, x_0, x_1)


conditioner = Conditioner()
flow_head = SimpleMLPAdaLN(2, 4, 2, 4, 2, 2)
flow_head.time_embed = nn.ModuleList(
    [TimestepEmbedder(4, frequency_embedding_size=4, max_period=100) for _ in range(2)]
)
transformer = StreamingTransformer(
    d_model=4,
    num_heads=2,
    num_layers=1,
    layer_scale=1.0,
    dim_feedforward=6,
    context=None,
    max_period=100.0,
)
flow_lm = FlowLMModel(
    conditioner=conditioner,
    flow_net=flow_head,
    transformer=transformer,
    dim=4,
    ldim=2,
    dtype=torch.float32,
    insert_bos_before_voice=True,
    flow_type="lsd",
)
flow_lm.speaker_proj_weight = nn.Parameter(torch.empty(4, 2))

with torch.no_grad():
    flow_lm.bos_emb.copy_(torch.tensor([0.15, -0.25]))
    flow_lm.bos_before_voice.copy_(torch.tensor([[[0.05, -0.1, 0.2, -0.3]]]))
    for i in range(conditioner.embed.weight.numel()):
        conditioner.embed.weight.flatten()[i] = ((i * 5) % 17 - 8) * 0.03
    for i in range(flow_lm.speaker_proj_weight.numel()):
        flow_lm.speaker_proj_weight.flatten()[i] = 0.02 + ((i * 7) % 13 - 6) * 0.025
fill_linear(flow_lm.input_linear, -0.015)
fill_linear(flow_lm.out_eos, 0.01)
layer = flow_lm.transformer.layers[0]
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
    flow_lm.out_norm.weight.copy_(torch.tensor([1.02, 0.98, 1.08, 0.92]))
    flow_lm.out_norm.bias.copy_(torch.tensor([0.01, -0.02, 0.03, -0.04]))
for i, embedding in enumerate(flow_head.time_embed):
    fill_linear(embedding.mlp[0], [0.01, -0.015][i])
    fill_linear(embedding.mlp[2], [-0.03, 0.025][i])
    with torch.no_grad():
        embedding.mlp[3].alpha.copy_(
            torch.tensor([[0.8, 1.1, 0.9, 1.2], [1.05, 0.95, 1.15, 0.85]][i])
        )
fill_linear(flow_head.input_proj, 0.03)
fill_linear(flow_head.cond_embed, -0.02)
block_specs = [
    (0.02, -0.01, 0.005, [1.1, 0.9, 1.2, 0.8], [0.02, -0.03, 0.01, 0.04]),
    (-0.025, 0.015, -0.004, [0.95, 1.05, 0.85, 1.15], [-0.01, 0.02, -0.04, 0.03]),
]
for block, spec in zip(flow_head.res_blocks, block_specs):
    fill_linear(block.mlp[0], spec[0])
    fill_linear(block.mlp[2], spec[1])
    fill_linear(block.adaLN_modulation[1], spec[2])
    with torch.no_grad():
        block.in_ln.weight.copy_(torch.tensor(spec[3]))
        block.in_ln.bias.copy_(torch.tensor(spec[4]))
fill_linear(flow_head.final_layer.linear, 0.02)
fill_linear(flow_head.final_layer.adaLN_modulation[1], -0.006)

objective = DeterministicLSD()
fill_linear(objective.w_s_t[0], 0.02)
fill_linear(objective.w_s_t[2], -0.01)
fill_linear(objective.w_s_t[4], 0.015)
logvar_outputs = []

def capture_logvar(_module, _inputs, output):
    output.retain_grad()
    logvar_outputs.append(output)

objective.w_s_t.register_forward_hook(capture_logvar)
args = TrainArgs()
args.flow_batch_multiplier = 1
args.eos_loss_weight = 0.1
args.text_dropout = 0.0
args.voice_dropout = 0.0
model = TrainableTTS(flow_lm, objective, args).train()

latents = torch.tensor(
    [[[0.2, -0.4], [0.6, 0.1], [-0.3, 0.7]]], dtype=torch.float32, requires_grad=True
)
voice = torch.tensor(
    [[[-0.2, 0.5], [0.8, -0.1]]], dtype=torch.float32, requires_grad=True
)
mask = torch.tensor([[True, True, False]])
text_tokens = [torch.tensor([1, 3], dtype=torch.long)]
# TrainableTTS.forward draws two zero-dropout keep decisions before randn_like.
torch.manual_seed(1234)
loss, metrics = model(
    latents,
    mask,
    text_tokens,
    voice,
    num_voice_prompt_frames=torch.tensor([2]),
)
loss.backward()
noise = objective.recorded_noise
assert noise is not None and noise.shape == (2, 2)
assert len(logvar_outputs) == 2
(diagonal_logvar, distill_logvar) = logvar_outputs
assert diagonal_logvar.grad is not None and distill_logvar.grad is not None
noise_padded = torch.cat([noise, torch.zeros(1, 2)], dim=0)

parameters = {
    name: value.detach().flatten().tolist() for name, value in model.state_dict().items()
}
gradients = {
    name: value.grad.detach().flatten().tolist()
    for name, value in model.named_parameters()
}
trainable = dict(model.named_parameters())
before_step = {
    name: value.detach().flatten().tolist() for name, value in trainable.items()
}
optimizer = torch.optim.AdamW(
    list(trainable.values()),
    lr=0.001,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)
optimizer.step()
after_step = {
    name: value.detach().flatten().tolist() for name, value in trainable.items()
}
ema_after_step = {
    name: [0.9 * old + 0.1 * new for old, new in zip(before_step[name], after_step[name])]
    for name in before_step
}
fixture = {
    "schema": 1,
    "generator": f"pocket-tts .venv/bin/python (torch {torch.__version__})",
    "upstream_revision": REVISION,
    "seed": 1234,
    "frames": 3,
    "voice_frames": 2,
    "hidden": 4,
    "latent_dim": 2,
    "vocabulary": 6,
    "config": {"p_equal": 0.75, "eos_loss_weight": 0.1, "flow_batch_multiplier": 1},
    "metrics": {name: value.detach().item() for name, value in metrics.items()},
    "normalized_latents": latents.detach().flatten().tolist(),
    "voice_latents": voice.detach().flatten().tolist(),
    "text_tokens": text_tokens[0].tolist(),
    "mask": mask.flatten().tolist(),
    "noise": noise_padded.flatten().tolist(),
    "diagonal_time": torch.cat([objective.diagonal_times.flatten(), torch.tensor([0.0])]).tolist(),
    "distill_s": torch.cat([objective.start_times.flatten(), torch.tensor([0.0])]).tolist(),
    "distill_t": torch.cat([objective.end_times.flatten(), torch.tensor([0.0])]).tolist(),
    "diagonal_log_variance": diagonal_logvar.detach().flatten().tolist(),
    "distill_log_variance": distill_logvar.detach().flatten().tolist(),
    "d_diagonal_log_variance": diagonal_logvar.grad.detach().flatten().tolist(),
    "d_distill_log_variance": distill_logvar.grad.detach().flatten().tolist(),
    "d_normalized_latents": latents.grad.flatten().tolist(),
    "d_voice_latents": voice.grad.flatten().tolist(),
    "parameters": parameters,
    "gradients": gradients,
    "adamw": {
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "parameters_after_step": after_step,
        "ema_decay": 0.9,
        "ema_after_step": ema_after_step,
    },
}
print(json.dumps(fixture, indent=2))
