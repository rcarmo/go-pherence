#!/usr/bin/env python3
"""Audit same-position sibling and question changes on the approved MoJev scorer.

This deliberately records a possible isolation failure rather than asserting
that a mask-only test guarantees model-level independence.
"""
import hashlib
import inspect
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
MODEL_SHA = "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458"
CONFIG_SHA = "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9"
WEIGHTS_SHA = "eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50"
WEIGHTS_SIZE = 1710234304
TRANSFORMERS_SHA = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: mojev_oracle_released_isolation.py PINNED_SOURCE CHECKPOINT_DIR OUTPUT.json")
    source, checkpoint, dest = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    for path, expected in ((source / "mojev/modeling.py", MODEL_SHA), (checkpoint / "config.json", CONFIG_SHA)):
        if digest(path) != expected:
            raise SystemExit(f"asset hash mismatch: {path}")
    weights = checkpoint / "model.safetensors"
    if weights.stat().st_size != WEIGHTS_SIZE or digest(weights) != WEIGHTS_SHA:
        raise SystemExit("weight size/hash mismatch")
    sys.path.insert(0, str(source))
    import torch
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    from mojev.modeling import PackedScorer
    if transformers.__version__ != "5.17.0" or digest(pathlib.Path(inspect.getfile(qwen))) != TRANSFORMERS_SHA:
        raise SystemExit("Transformers implementation mismatch")
    torch.set_num_threads(2)
    model = PackedScorer.from_pretrained(str(checkpoint), local_files_only=True, use_safetensors=True).eval()
    length = 12
    state = torch.tensor([[1, 1, 1, 1] + [0] * 8], dtype=torch.float32)
    questions = torch.zeros(1, 2, length)
    questions[0, 0, 4:6] = 1
    questions[0, 1, 8:10] = 1
    candidates = torch.zeros(1, 2, 2, length)
    for field, option, position in ((0, 0, 6), (0, 1, 7), (1, 0, 10), (1, 1, 11)):
        candidates[0, field, option, position] = 1
    base = [10, 15, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62]
    sibling = base.copy()
    sibling[7] = 123
    other_question = base.copy()
    other_question[8] = 124
    batch = dict(packed_mask=torch.ones(1, length, dtype=torch.bool),
                 context_span=state, field_span=questions, option_span=candidates,
                 option_mask=torch.ones(1, 2, 2, dtype=torch.bool))
    mask = model.build_mask(state, questions, candidates)
    if mask[0, 0, 10, 7].item() == 0 or mask[0, 0, 6, 8].item() == 0:
        raise SystemExit("tree mask unexpectedly allows cross-question access")
    with torch.inference_mode():
        outputs = [model(dict(batch, packed_ids=torch.tensor([ids], dtype=torch.long)))[0].tolist()
                   for ids in (base, sibling, other_question)]
    result = dict(schema=1, source_revision=REV, modeling_sha256=MODEL_SHA,
                  transformers_version=transformers.__version__, transformers_qwen_sha256=TRANSFORMERS_SHA,
                  config_sha256=CONFIG_SHA, weights_sha256=WEIGHTS_SHA, weights_size=WEIGHTS_SIZE,
                  base_ids=base, changes=[dict(name="sibling_candidate", position=7, token_id=123),
                                          dict(name="other_question", position=8, token_id=124)],
                  logits=dict(base=outputs[0], sibling_candidate=outputs[1], other_question=outputs[2]))
    dest.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
