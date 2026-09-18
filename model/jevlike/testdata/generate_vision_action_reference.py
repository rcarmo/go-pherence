"""Generate VisionActionScorer parity data from upstream MIT-licensed Jevlike.

Run with:
  PYTHONPATH=/tmp/jevlike .venv-speaker/bin/python \
    projects/go-pherence/model/jevlike/testdata/generate_vision_action_reference.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType

import torch

from jevlike.vision import DoomScorerV2, position_2d


def export_head(head) -> dict[str, object]:
    return {
        "width": int(head.query.weight.shape[1]),
        "rank": int(head.rank),
        "context_norm_weight": head.context_norm.weight.detach().tolist(),
        "context_norm_bias": head.context_norm.bias.detach().tolist(),
        "option_norm_weight": head.option_norm.weight.detach().tolist(),
        "option_norm_bias": head.option_norm.bias.detach().tolist(),
        "query_weight": head.query.weight.detach().flatten().tolist(),
        "key_weight": head.key.weight.detach().flatten().tolist(),
        "value_weight": head.value.weight.detach().flatten().tolist(),
    }


def main() -> None:
    torch.manual_seed(20260918)
    width = 4
    rank = 4
    reads = 2
    actions = 5
    patch_rows = 2
    patch_columns = 3
    patches = patch_rows * patch_columns
    batch = 2
    options_per_row = 3

    model = DoomScorerV2(actions=actions, width=width, rank=rank, reads=reads).float().eval()

    raw_context = torch.tensor(
        [
            [
                [0.10, -0.20, 0.30, 0.50],
                [1.10, -0.40, 0.20, -1.00],
                [0.00, 0.70, -0.60, 0.90],
                [-1.20, 0.30, 0.80, -0.50],
                [0.60, -0.70, 1.40, 0.20],
                [0.40, 0.90, -1.10, -0.30],
            ],
            [
                [-0.80, 0.50, 0.10, 1.20],
                [0.30, -1.00, 0.90, -0.40],
                [1.50, 0.20, -0.30, 0.70],
                [-0.60, -0.20, 0.40, 1.10],
                [0.80, 1.30, -0.90, 0.00],
                [-1.40, 0.60, 0.50, -0.70],
            ],
        ],
        dtype=torch.float32,
    )
    if tuple(raw_context.shape) != (batch, patches, width):
        raise RuntimeError(f"unexpected raw_context shape {tuple(raw_context.shape)}")

    option_ids = torch.tensor(
        [
            [4, 1, 0],
            [2, 3, 1],
        ],
        dtype=torch.long,
    )
    if tuple(option_ids.shape) != (batch, options_per_row):
        raise RuntimeError(f"unexpected option_ids shape {tuple(option_ids.shape)}")

    model.positions = position_2d(patch_rows, patch_columns, width).float()

    def encode_override(self, observations: torch.Tensor):
        features = raw_context.to(device=observations.device, dtype=torch.float32)
        positions = self.positions.to(device=observations.device, dtype=features.dtype)
        return features, positions.unsqueeze(0).expand(features.shape[0], -1, -1)

    model.encode = MethodType(encode_override, model)
    observations = torch.zeros(batch, 4, 1, 1, dtype=torch.float32)

    with torch.inference_mode():
        logits, values, trace = model.forward_trace(observations, option_ids)
    attention = trace["attention_map"].float()
    entropy = (-(attention * attention.clamp_min(1e-12).log()).sum(-1) / math.log(attention.shape[-1])).detach()

    data = {
        "schema": 1,
        "absolute_tolerance": 2e-5,
        "upstream_source": "jevlike/vision.py:DoomScorerV2._forward (MIT)",
        "torch_version": torch.__version__,
        "config": {
            "width": width,
            "rank": rank,
            "actions": actions,
            "reads": reads,
            "patch_rows": patch_rows,
            "patch_columns": patch_columns,
        },
        "positions": model.positions.detach().flatten().tolist(),
        "option_embedding": model.options.weight.detach().flatten().tolist(),
        "heads": [export_head(model.head), *(export_head(head) for head in model.extra_heads)],
        "value_norm_weight": model.value_head[0].weight.detach().flatten().tolist(),
        "value_norm_bias": model.value_head[0].bias.detach().flatten().tolist(),
        "value_weight": model.value_head[1].weight.detach().flatten().tolist(),
        "value_bias": float(model.value_head[1].bias.detach().item()),
        "raw_context": raw_context.tolist(),
        "option_ids": option_ids.tolist(),
        "forward": {
            "logits": logits.detach().tolist(),
            "values": values.detach().tolist(),
        },
        "trace": {
            "query_matrix": trace["query_matrix"].tolist(),
            "key_matrix": trace["key_matrix"].tolist(),
            "value_matrix": trace["value_matrix"].tolist(),
            "attention_map": trace["attention_map"].tolist(),
            "logits_matrix": trace["logits_matrix"].tolist(),
            "probabilities": trace["probabilities"].tolist(),
            "entropy": entropy.tolist(),
        },
    }

    output = Path(__file__).with_name("vision_action_reference.json")
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
