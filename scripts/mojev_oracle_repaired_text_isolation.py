#!/usr/bin/env python3
"""Pin a separate, branch-local MoJev text-only repair observation.

Each candidate gets its own state/question/candidate forward with original
absolute text positions and a per-branch bidirectional tree mask for full
attention. Linear attention has only that branch's recurrent history. This is
an experimental repaired policy, not parity with the released packed forward.
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
        raise SystemExit("usage: mojev_oracle_repaired_text_isolation.py SOURCE CHECKPOINT OUTPUT.json")
    source, checkpoint, dest = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    for path, expected in ((source / "mojev/modeling.py", MODEL_SHA), (checkpoint / "config.json", CONFIG_SHA)):
        if digest(path) != expected:
            raise SystemExit(f"asset hash mismatch: {path}")
    weights = checkpoint / "model.safetensors"
    if weights.stat().st_size != WEIGHTS_SIZE or digest(weights) != WEIGHTS_SHA:
        raise SystemExit("weights mismatch")
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
    base = [10, 15, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62]
    cases = {"base": base, "sibling_candidate": base[:7] + [123] + base[8:],
             "other_question": base[:8] + [124] + base[9:]}
    state = list(range(4))
    questions = [[4, 5], [8, 9]]
    candidates = [[[6], [7]], [[10], [11]]]

    def branch(ids, question, candidate):
        positions = state + question + candidate
        n_state, n_question = len(state), len(question)
        width = len(positions)
        token_ids = torch.tensor([[ids[p] for p in positions]], dtype=torch.long)
        state_span = torch.zeros((1, width), dtype=torch.float32)
        question_span = torch.zeros((1, 1, width), dtype=torch.float32)
        candidate_span = torch.zeros((1, 1, 1, width), dtype=torch.float32)
        state_span[:, :n_state] = 1
        question_span[:, :, n_state:n_state + n_question] = 1
        candidate_span[:, :, :, n_state + n_question:] = 1
        mask = model.build_mask(state_span, question_span, candidate_span)
        # All four channels are text positions. Keep each token's original
        # position even though sibling tokens have been removed from the branch.
        position_ids = torch.tensor(positions, dtype=torch.long)[None, None, :].expand(4, 1, width)
        with torch.inference_mode():
            hidden = model.encoder(input_ids=token_ids, attention_mask=mask,
                                   position_ids=position_ids).last_hidden_state.float()
            hidden = model.norm(hidden)
            context = hidden[:, :n_state].mean(1)
            field = hidden[:, n_state:n_state + n_question].mean(1)
            option = hidden[:, n_state + n_question:].mean(1)
            query = model.context_proj(context) + model.context_proj(field)
            key = model.option_proj(option)
            return float(((query * key).sum(-1) / model.rank ** 0.5)[0])

    outputs = {name: [[branch(ids, q, c) for c in options]
                      for q, options in zip(questions, candidates)] for name, ids in cases.items()}
    if outputs["base"][1] != outputs["sibling_candidate"][1] or outputs["base"][0][0] != outputs["sibling_candidate"][0][0]:
        raise SystemExit("sibling leak remains")
    if outputs["base"][0] != outputs["other_question"][0]:
        raise SystemExit("cross-question leak remains")
    if outputs["base"][0][1] == outputs["sibling_candidate"][0][1] or outputs["base"][1] == outputs["other_question"][1]:
        raise SystemExit("substitutions failed to affect their own branches")
    dest.write_text(json.dumps(dict(schema=1, source_revision=REV, modeling_sha256=MODEL_SHA,
                                    transformers_version=transformers.__version__, transformers_qwen_sha256=TRANSFORMERS_SHA,
                                    config_sha256=CONFIG_SHA, weights_sha256=WEIGHTS_SHA, weights_size=WEIGHTS_SIZE,
                                    policy="branch-local-text-absolute-positions-bidirectional-full-causal-linear",
                                    base_ids=base, state_positions=state, question_positions=questions,
                                    candidate_positions=candidates, changed_ids={k: v for k, v in cases.items() if k != "base"},
                                    logits=outputs), separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
