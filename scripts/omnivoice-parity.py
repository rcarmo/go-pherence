#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
from pathlib import Path

import torch
import transformers
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

SEED = 123
SEQ_LEN = 3
HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 12
NUM_ATTENTION_HEADS = 2
NUM_KEY_VALUE_HEADS = 1
HEAD_DIM = 4
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0
LINEAR_SCALE = 0.15
NORM_SCALE = 0.15
INPUT_SCALE = 0.10
MASK_BLOCK_VALUE = -1.0e9
ATTN_IMPL = "eager"
OUT_PATH = Path("testdata/omnivoice/tiny-block.json")

WEIGHT_ORDER = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def tensor_payload(t: torch.Tensor, *, squeeze_batch: bool = False) -> dict[str, object]:
    x = t.detach().to(torch.float32).cpu().contiguous()
    if squeeze_batch:
        if x.ndim == 0 or x.shape[0] != 1:
            raise ValueError(f"cannot squeeze batch from shape {tuple(x.shape)}")
        x = x[0]
    return {
        "shape": list(x.shape),
        "data": [float(v) for v in x.reshape(-1).tolist()],
    }



def assign_weights(layer: Qwen3DecoderLayer) -> dict[str, dict[str, object]]:
    params = dict(layer.named_parameters())
    weights: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for name in WEIGHT_ORDER:
            param = params[name]
            if "norm" in name:
                value = 1.0 + torch.randn_like(param) * NORM_SCALE
            else:
                value = torch.randn_like(param) * LINEAR_SCALE
            param.copy_(value)
            weights[name] = tensor_payload(param)
    return weights



def build_fixture() -> dict[str, object]:
    torch.manual_seed(SEED)

    config = Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=1,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=RMS_NORM_EPS,
        max_position_embeddings=16,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        rope_parameters={"rope_type": "default", "rope_theta": ROPE_THETA},
    )
    config._attn_implementation = ATTN_IMPL

    layer = Qwen3DecoderLayer(config, layer_idx=0)
    rotary = Qwen3RotaryEmbedding(config)
    layer.eval()
    rotary.eval()

    weights = assign_weights(layer)

    hidden_states = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float32) * INPUT_SCALE
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(hidden_states, position_ids)

    attention_mask = torch.zeros((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.float32)
    attention_mask[:, :, :, 2] = MASK_BLOCK_VALUE

    with torch.no_grad():
        output_unmasked = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        output_masked = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    return {
        "schema": "go-pherence/omnivoice-qwen3-decoder-block/v1",
        "generator": {
            "script": "scripts/omnivoice-parity.py",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "seed": SEED,
            "dtype": "float32",
            "device": "cpu",
            "attention_implementation": ATTN_IMPL,
            "module": "transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer",
            "linear_weight_scale": LINEAR_SCALE,
            "norm_weight_scale": NORM_SCALE,
            "input_scale": INPUT_SCALE,
        },
        "config": {
            "dimensions": {
                "batch_size": 1,
                "seq_len": SEQ_LEN,
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_attention_heads": NUM_ATTENTION_HEADS,
                "num_key_value_heads": NUM_KEY_VALUE_HEADS,
                "head_dim": HEAD_DIM,
            },
            "epsilon": RMS_NORM_EPS,
            "theta": ROPE_THETA,
        },
        "position_ids": {
            "shape": list(position_ids.shape),
            "data": [int(v) for v in position_ids.reshape(-1).tolist()],
        },
        "input": tensor_payload(hidden_states, squeeze_batch=True),
        "weights": weights,
        "attention_mask_block_key2": {
            "shape": list(attention_mask.shape),
            "blocked_key_position": 2,
            "semantics": "additive attention mask; 0.0 allows attention, -1e9 blocks key position 2 for every query",
            "data": [float(v) for v in attention_mask.reshape(-1).tolist()],
        },
        "outputs": {
            "unmasked": tensor_payload(output_unmasked, squeeze_batch=True),
            "mask_block_key2": tensor_payload(output_masked, squeeze_batch=True),
        },
    }



def main() -> None:
    fixture = build_fixture()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"schema={fixture['schema']}")
    print(f"input_shape={fixture['input']['shape']} unmasked_shape={fixture['outputs']['unmasked']['shape']}")


if __name__ == "__main__":
    main()
