#!/usr/bin/env python3
"""Bounded CPU readout oracle from an approved, hash-pinned MoJev checkpoint.

Run against the pinned MIT source; no tokenizer or external model download occurs.
Writes pre-LayerNorm encoder hidden rows, spans, option mask and scorer logits.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
MODEL_SHA = "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458"
CONFIG_SHA = "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9"
WEIGHTS_SHA = "eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50"
WEIGHTS_SIZE = 1710234304


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: mojev_oracle_released_head.py PINNED_SOURCE CHECKPOINT_DIR OUTPUT.json")
    source, checkpoint, dest = map(lambda x: pathlib.Path(x).resolve(), sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    if digest(source / "mojev/modeling.py") != MODEL_SHA:
        raise SystemExit("source hash mismatch")
    if digest(checkpoint / "config.json") != CONFIG_SHA:
        raise SystemExit("config hash mismatch")
    weights = checkpoint / "model.safetensors"
    if weights.stat().st_size != WEIGHTS_SIZE or digest(weights) != WEIGHTS_SHA:
        raise SystemExit("weight size/hash mismatch")
    sys.path.insert(0, str(source))
    import torch
    from mojev.modeling import PackedScorer

    torch.set_num_threads(2)
    model = PackedScorer.from_pretrained(str(checkpoint), local_files_only=True, use_safetensors=True).eval()
    if next(model.encoder.parameters()).dtype != torch.bfloat16 or model.context_proj.weight.dtype != torch.float32:
        raise SystemExit("checkpoint dtype mismatch")
    ids = [10, 15, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62]
    state = [1, 1, 1, 1] + [0] * 8
    questions = [[0] * 12 for _ in range(2)]
    questions[0][4:6] = [1, 1]
    questions[1][8:10] = [1, 1]
    candidates = [[[0] * 12 for _ in range(2)] for _ in range(2)]
    for f, n, pos in ((0, 0, 6), (0, 1, 7), (1, 0, 10), (1, 1, 11)):
        candidates[f][n][pos] = 1
    option_mask = [[1, 1], [1, 0]]
    batch = dict(packed_ids=torch.tensor([ids]), packed_mask=torch.ones(1, 12, dtype=torch.bool),
                 context_span=torch.tensor([state], dtype=torch.float32),
                 field_span=torch.tensor([questions], dtype=torch.float32),
                 option_span=torch.tensor([candidates], dtype=torch.float32),
                 option_mask=torch.tensor([option_mask], dtype=torch.bool))
    saved = []
    def capture(_module, _args, output):
        saved.append(output.last_hidden_state.detach().float().cpu().tolist()[0])
    hook = model.encoder.register_forward_hook(capture)
    with torch.inference_mode():
        logits = model(batch).cpu()[0]
    hook.remove()
    if len(saved) != 1 or len(saved[0]) != 12 or len(saved[0][0]) != 1024:
        raise SystemExit("unexpected encoder hidden dimensions")
    result = dict(schema=1, source_revision=REV, modeling_sha256=MODEL_SHA,
                  config_sha256=CONFIG_SHA, weights_sha256=WEIGHTS_SHA,
                  weights_size=WEIGHTS_SIZE, token_ids=ids, state=state,
                  questions=questions, candidates=candidates, option_mask=option_mask,
                  pre_norm_hidden=saved[0], logits=logits.tolist())
    dest.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
