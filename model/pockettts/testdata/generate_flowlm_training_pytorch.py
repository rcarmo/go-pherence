#!/usr/bin/env python3
"""Generate flowlm_training_pytorch.json from pinned upstream training layout."""

import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.text_conditioner import LUTConditioner
from pocket_tts.modules.transformer import StreamingTransformer
from training.modules.conditioner import build_sequences_with_conditions
from training.modules.model import TrainableTTS

UPSTREAM_REVISION = "0acce6b2f390150267557770d2098c5caa9a18ac"
checkout = Path.cwd().resolve()
if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() != UPSTREAM_REVISION:
    raise SystemExit("wrong upstream revision")
imports = {
    Path(inspect.getfile(build_sequences_with_conditions)).resolve(): checkout / "training/modules/conditioner.py",
    Path(inspect.getfile(TrainableTTS)).resolve(): checkout / "training/modules/model.py",
    Path(inspect.getfile(StreamingTransformer)).resolve(): checkout / "pocket_tts/modules/transformer.py",
}
for actual, expected in imports.items():
    if actual != expected: raise SystemExit(f"imported {actual}, want {expected}")
subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "training/modules/conditioner.py", "training/modules/model.py", "pocket_tts/modules/transformer.py", "pocket_tts/modules/attention.py", "pocket_tts/modules/rope.py", "pocket_tts/modules/layer_scale.py"], cwd=checkout, check=True)


def fill_linear(linear: nn.Linear, start: float) -> None:
    with torch.no_grad():
        flat = linear.weight.flatten()
        for i in range(flat.numel()): flat[i] = start + ((i * 7) % 13 - 6) * 0.025
        if linear.bias is not None:
            for i in range(linear.bias.numel()): linear.bias[i] = start / 3 + (i - 1) * 0.02


class FakeConditioner(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__(); self.embed = nn.Embedding(vocab, hidden)
    def forward(self, tokens): return self.embed(tokens)


class FakeFlowLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioner = FakeConditioner(6, 4)
        self.bos_emb = nn.Parameter(torch.tensor([0.15, -0.25]))
        self.bos_before_voice = nn.Parameter(torch.tensor([[[0.05, -0.1, 0.2, -0.3]]]))
        self.speaker_proj_weight = nn.Parameter(torch.empty(4, 2))
        self.input_linear = nn.Linear(2, 4, bias=False)
        self.transformer = StreamingTransformer(d_model=4, num_heads=2, num_layers=1, layer_scale=1.0, dim_feedforward=6, context=None, max_period=100.0)
        self.out_norm = nn.LayerNorm(4, eps=1e-5)
        self.out_eos = nn.Linear(4, 1)


fl = FakeFlowLM()
with torch.no_grad():
    for i in range(fl.conditioner.embed.weight.numel()): fl.conditioner.embed.weight.flatten()[i] = ((i * 5) % 17 - 8) * 0.03
fill_linear(fl.input_linear, -0.015)
fill_linear(fl.out_eos, 0.01)
with torch.no_grad():
    flat = fl.speaker_proj_weight.flatten()
    for i in range(flat.numel()): flat[i] = 0.02 + ((i * 7) % 13 - 6) * 0.025
layer = fl.transformer.layers[0]
fill_linear(layer.self_attn.in_proj, 0.01); fill_linear(layer.self_attn.out_proj, -0.02); fill_linear(layer.linear1, 0.03); fill_linear(layer.linear2, -0.015)
with torch.no_grad():
    layer.norm1.weight.copy_(torch.tensor([1.1, .9, 1.2, .8])); layer.norm1.bias.copy_(torch.tensor([.02, -.03, .01, .04]))
    layer.norm2.weight.copy_(torch.tensor([.95, 1.05, .85, 1.15])); layer.norm2.bias.copy_(torch.tensor([-.01, .02, -.04, .03]))
    layer.layer_scale_1.scale.copy_(torch.tensor([.8, 1.1, .9, 1.2])); layer.layer_scale_2.scale.copy_(torch.tensor([1.05, .95, 1.15, .85]))
    fl.out_norm.weight.copy_(torch.tensor([1.02, .98, 1.08, .92])); fl.out_norm.bias.copy_(torch.tensor([.01, -.02, .03, -.04]))

latents = torch.tensor([[[.2, -.4], [.6, .1], [-.3, .7]]], requires_grad=True)
voice = torch.tensor([[[-.2, .5], [.8, -.1]]], requires_grad=True)
tokens = [torch.tensor([1, 3], dtype=torch.long)]
args = SimpleNamespace(voice_dropout=0.0, text_dropout=0.0)
sequence, prefix = build_sequences_with_conditions(args, latents, tokens, voice, cfg_dropout=False, fl=fl, num_voice_prompt_frames=torch.tensor([2]))
transformed = fl.out_norm(fl.transformer(sequence, model_state=None))
idx = prefix[:, None] + torch.arange(latents.shape[1])[None, :]
z = transformed.gather(1, idx[:, :, None].expand(-1, -1, transformed.shape[-1]))
eos = fl.out_eos(z).squeeze(-1)
d_z = torch.tensor([[[.1, -.2, .3, -.4], [.5, -.6, .7, -.8], [-.3, .4, -.5, .6]]])
d_eos = torch.tensor([[.2, -.35, .45]])
loss = (z * d_z).sum() + (eos * d_eos).sum(); loss.backward()

fixture = {
    "schema": 1, "generator": f"pocket-tts .venv/bin/python (torch {torch.__version__})", "upstream_revision": UPSTREAM_REVISION,
    "frames": 3, "voice_frames": 2, "hidden": 4, "latent_dim": 2, "vocabulary": 6,
    "normalized_latents": latents.detach().flatten().tolist(), "voice_latents": voice.detach().flatten().tolist(), "text_tokens": tokens[0].tolist(),
    "d_z": d_z.flatten().tolist(), "d_eos": d_eos.flatten().tolist(), "prefix_rows": prefix.item(), "sequence": sequence.detach().flatten().tolist(),
    "z": z.detach().flatten().tolist(), "eos": eos.detach().flatten().tolist(), "d_normalized_latents": latents.grad.flatten().tolist(), "d_voice_latents": voice.grad.flatten().tolist(),
    "parameters": {name: value.detach().flatten().tolist() for name, value in fl.state_dict().items()},
    "gradients": {name: value.grad.detach().flatten().tolist() for name, value in fl.named_parameters()},
}
print(json.dumps(fixture, indent=2))
