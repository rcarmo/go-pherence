#!/usr/bin/env /workspace/projects/spock-tts/.venv/bin/python
"""Generate deterministic OmniVoice sampling fixtures.

This script imports the installed upstream helper functions where they are
standalone and deterministic:
  - omnivoice.models.omnivoice._get_time_steps
  - omnivoice.models.omnivoice._filter_top_k
  - omnivoice.models.omnivoice._gumbel_sample

For the surrounding control flow we use exact formula excerpts from the same
file because the upstream implementations either live inside OmniVoice methods
or draw randomness internally without accepting caller-provided noise:
  - _predict_tokens_with_scoring: exact log_softmax / CFG / mask / optional
    top-k + gumbel excerpt from OmniVoice._predict_tokens_with_scoring
  - confidence selection and token updates: exact score penalty / optional
    gumbel / mask / top-k / scatter excerpt from OmniVoice._generate_iterative

The goal is stable Go-only parity tests. We inject explicit uniforms into the
upstream _gumbel_sample helper by temporarily monkey-patching torch.rand_like.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from omnivoice.models.omnivoice import _filter_top_k, _get_time_steps, _gumbel_sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "omnivoice" / "sampler.json"

CODEBOOKS = 3
TARGET_LEN = 4
VOCAB = 20
MASK_ID = 19
NUM_STEP = 6
GUIDANCE_SCALE = 1.5
T_SHIFT = 0.1
LAYER_PENALTY = 5.0
POSITION_TEMPERATURE = 1.3
CLASS_TEMPERATURE = 0.7
TOP_K_RATIO = 0.1
SELECTION_K = 5
SCHEDULE_TARGET_LENS = [1, 4, 7]


def encode_floats(t: torch.Tensor):
    out = []
    for v in t.flatten().tolist():
        if math.isinf(v):
            out.append("-inf" if v < 0 else "+inf")
        elif math.isnan(v):
            out.append("nan")
        else:
            out.append(v)
    return out


@contextmanager
def patched_rand_like(replacement: torch.Tensor):
    orig = torch.rand_like
    torch.rand_like = lambda x: replacement.to(dtype=x.dtype, device=x.device)
    try:
        yield
    finally:
        torch.rand_like = orig


def deterministic_logits() -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(CODEBOOKS * TARGET_LEN * VOCAB, dtype=torch.float32).reshape(
        1, CODEBOOKS, TARGET_LEN, VOCAB
    )
    cond = (
        0.35 * torch.sin(idx / 3.0)
        + 0.17 * torch.cos(idx / 5.0)
        + 0.03 * torch.remainder(idx, 7.0)
    )
    uncond = (
        0.28 * torch.cos((idx + 2.0) / 4.0)
        - 0.11 * torch.sin((idx + 1.0) / 6.0)
        - 0.02 * torch.remainder(idx, 5.0)
    )
    cond[..., MASK_ID] += 3.75
    uncond[..., MASK_ID] += 1.25
    return cond, uncond


def deterministic_uniforms(count: int, mul: int, add: int) -> torch.Tensor:
    vals = [(((i * mul + add) % 997) + 0.5) / 997.5 for i in range(count)]
    return torch.tensor(vals, dtype=torch.float32)


def exact_predict_tokens_with_scoring(
    cond_logits: torch.Tensor,
    uncond_logits: torch.Tensor,
    class_uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    c_log_probs = F.log_softmax(cond_logits, dim=-1)
    u_log_probs = F.log_softmax(uncond_logits, dim=-1)
    guided = torch.log_softmax(
        c_log_probs + GUIDANCE_SCALE * (c_log_probs - u_log_probs), dim=-1
    )
    guided[..., MASK_ID] = -float("inf")
    filtered = _filter_top_k(guided, ratio=TOP_K_RATIO)
    with patched_rand_like(class_uniforms):
        class_scores = _gumbel_sample(filtered, CLASS_TEMPERATURE)
    pred_tokens = class_scores.argmax(dim=-1)
    confidence = guided.max(dim=-1).values
    return guided, filtered, class_scores, pred_tokens, confidence


def exact_position_update(
    tokens_before: torch.Tensor,
    pred_tokens: torch.Tensor,
    confidence: torch.Tensor,
    position_uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    layer_ids = torch.arange(CODEBOOKS, dtype=torch.float32).view(1, CODEBOOKS, 1)
    scores = confidence - layer_ids * LAYER_PENALTY
    with patched_rand_like(position_uniforms):
        gumbel_scores = _gumbel_sample(scores, POSITION_TEMPERATURE)
    masked_scores = gumbel_scores.clone()
    masked_scores.masked_fill_(tokens_before != MASK_ID, -float("inf"))
    _, topk_idx = torch.topk(masked_scores.flatten(), SELECTION_K)
    flat_tokens = tokens_before.flatten().clone()
    flat_pred = pred_tokens.flatten()
    flat_tokens[topk_idx] = flat_pred[topk_idx]
    return scores, gumbel_scores, topk_idx.tolist(), flat_tokens.view_as(tokens_before)


def main() -> None:
    cond_logits, uncond_logits = deterministic_logits()
    class_uniforms = deterministic_uniforms(cond_logits.numel(), 37, 11).view_as(cond_logits)
    position_uniforms = deterministic_uniforms(CODEBOOKS * TARGET_LEN, 53, 7).view(
        1, CODEBOOKS, TARGET_LEN
    )

    timesteps = _get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=NUM_STEP,
        t_shift=T_SHIFT,
    )
    schedules = []
    for target_len in SCHEDULE_TARGET_LENS:
        total_mask = target_len * CODEBOOKS
        rem = total_mask
        sched = []
        for step in range(NUM_STEP):
            if step == NUM_STEP - 1:
                num = rem
            else:
                num = min(
                    math.ceil(total_mask * float(timesteps[step + 1] - timesteps[step])),
                    rem,
                )
            sched.append(int(num))
            rem -= int(num)
        schedules.append(sched)

    guided, filtered, class_scores, pred_tokens, confidence = exact_predict_tokens_with_scoring(
        cond_logits, uncond_logits, class_uniforms
    )

    tokens_before = torch.tensor(
        [[[MASK_ID, MASK_ID, 5, MASK_ID], [MASK_ID, 7, MASK_ID, MASK_ID], [3, MASK_ID, MASK_ID, MASK_ID]]],
        dtype=torch.long,
    )
    position_scores, position_gumbel_scores, selected_flat_indices, tokens_after = exact_position_update(
        tokens_before, pred_tokens, confidence, position_uniforms
    )

    data = {
        "meta": {
            "python": str(Path(torch.__file__).resolve()),
            "source": "/workspace/projects/spock-tts/.venv/lib/python3.13/site-packages/omnivoice/models/omnivoice.py",
            "notes": [
                "time steps use upstream _get_time_steps",
                "class filtering uses upstream _filter_top_k",
                "gumbel perturbation uses upstream _gumbel_sample with patched torch.rand_like to inject deterministic uniforms",
                "classifier-free guidance and iterative selection follow exact excerpts from OmniVoice._predict_tokens_with_scoring and OmniVoice._generate_iterative",
            ],
        },
        "config": {
            "num_audio_codebook": CODEBOOKS,
            "target_len": TARGET_LEN,
            "audio_vocab_size": VOCAB,
            "audio_mask_id": MASK_ID,
            "num_step": NUM_STEP,
            "guidance_scale": GUIDANCE_SCALE,
            "t_shift": T_SHIFT,
            "layer_penalty_factor": LAYER_PENALTY,
            "position_temperature": POSITION_TEMPERATURE,
            "class_temperature": CLASS_TEMPERATURE,
            "class_top_k_ratio": TOP_K_RATIO,
            "selection_k": SELECTION_K,
        },
        "schedule": {
            "target_lens": SCHEDULE_TARGET_LENS,
            "timesteps": timesteps.tolist(),
            "schedules": schedules,
        },
        "cond_logits": {"shape": list(cond_logits.shape[1:]), "data": encode_floats(cond_logits)},
        "uncond_logits": {"shape": list(uncond_logits.shape[1:]), "data": encode_floats(uncond_logits)},
        "class_uniforms": {"shape": list(class_uniforms.shape[1:]), "data": encode_floats(class_uniforms)},
        "guided_log_probs": {"shape": list(guided.shape[1:]), "data": encode_floats(guided)},
        "filtered_log_probs": {"shape": list(filtered.shape[1:]), "data": encode_floats(filtered)},
        "class_gumbel_scores": {"shape": list(class_scores.shape[1:]), "data": encode_floats(class_scores)},
        "pred_tokens": {"shape": list(pred_tokens.shape[1:]), "data": pred_tokens.flatten().tolist()},
        "confidence_scores": {"shape": list(confidence.shape[1:]), "data": encode_floats(confidence)},
        "tokens_before": {"shape": list(tokens_before.shape[1:]), "data": tokens_before.flatten().tolist()},
        "position_uniforms": {"shape": list(position_uniforms.shape[1:]), "data": encode_floats(position_uniforms)},
        "position_scores": {"shape": list(position_scores.shape[1:]), "data": encode_floats(position_scores)},
        "position_gumbel_scores": {"shape": list(position_gumbel_scores.shape[1:]), "data": encode_floats(position_gumbel_scores)},
        "selected_flat_indices": selected_flat_indices,
        "tokens_after": {"shape": list(tokens_after.shape[1:]), "data": tokens_after.flatten().tolist()},
    }

    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
