#!/usr/bin/env python3
"""Independent F32 branch-local reference for native Go MoJev text execution.

Uses released weights, fresh state per candidate, local positions, bidirectional
full-attention nodes and causal linear attention. It deliberately differs from
both the released BF16 packed scorer and the earlier absolute-position probe.
"""
import copy
import hashlib
import inspect
import json
import pathlib
import subprocess
import sys

from mojev_oracle_repaired_text_isolation import (
    REV, MODEL_SHA, CONFIG_SHA, WEIGHTS_SHA, WEIGHTS_SIZE, TRANSFORMERS_SHA, digest,
)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: mojev_oracle_native_text.py SOURCE CHECKPOINT OUTPUT.json")
    source, checkpoint, dest = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source mismatch")
    for p, sha in ((source / "mojev/modeling.py", MODEL_SHA), (checkpoint / "config.json", CONFIG_SHA),
                   (checkpoint / "model.safetensors", WEIGHTS_SHA)):
        if digest(p) != sha:
            raise SystemExit(f"asset mismatch: {p}")
    if (checkpoint / "model.safetensors").stat().st_size != WEIGHTS_SIZE:
        raise SystemExit("weights size mismatch")
    sys.path.insert(0, str(source))
    import torch
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    from mojev.modeling import PackedScorer
    if transformers.__version__ != "5.17.0" or digest(pathlib.Path(inspect.getfile(qwen))) != TRANSFORMERS_SHA:
        raise SystemExit("Transformers mismatch")
    torch.set_num_threads(2)
    # Upcast the unchanged released BF16 tensor values, not new weights.
    model = PackedScorer.from_pretrained(str(checkpoint), local_files_only=True, use_safetensors=True).float().eval()
    base = dict(state=[10, 15, 21, 22], questions=[[31, 32], [51, 52]], candidates=[[[41], [42]], [[61], [62]]])
    cases = {"base": base}
    for name in ("sibling_token", "other_question", "sibling_length", "question_length", "candidate_order", "question_order", "long_nodes"):
        row = copy.deepcopy(base)
        if name == "sibling_token": row["candidates"][0][1] = [123]
        if name == "other_question": row["questions"][1][0] = 124
        if name == "sibling_length": row["candidates"][0][1] = [123, 124, 125, 126]
        if name == "question_length": row["questions"][0] += [71, 72, 73]
        if name == "candidate_order": row["candidates"][0].reverse(); row["candidates"][1].reverse()
        if name == "question_order": row["questions"].reverse(); row["candidates"].reverse()
        if name == "long_nodes":
            row = dict(state=list(range(10, 30)), questions=[list(range(30, 38)), list(range(70, 83))],
                       candidates=[[list(range(40, 52)), [60, 61]], [[84, 85, 86, 87], list(range(90, 109))]])
        cases[name] = row

    def run(row, capture):
        logits, hiddens = [], []
        for question, options in zip(row["questions"], row["candidates"]):
            scores = []
            for candidate in options:
                ns, nq = len(row["state"]), len(question)
                ids = row["state"] + question + candidate
                n = len(ids)
                state = torch.zeros((1, n)); state[:, :ns] = 1
                field = torch.zeros((1, 1, n)); field[:, :, ns:ns+nq] = 1
                option = torch.zeros((1, 1, 1, n)); option[:, :, :, ns+nq:] = 1
                batch = dict(packed_ids=torch.tensor([ids]), packed_mask=torch.ones((1, n), dtype=torch.bool),
                             context_span=state, field_span=field, option_span=option,
                             option_mask=torch.ones((1, 1, 1), dtype=torch.bool))
                captured = []
                hook = model.encoder.register_forward_hook(lambda m, a, out: captured.append(out.last_hidden_state.detach().float().clone())) if capture else None
                try:
                    with torch.inference_mode():
                        scores.append(float(model(batch)[0, 0, 0]))
                finally:
                    if hook is not None: hook.remove()
                if capture: hiddens.append(captured[0][0].tolist())
            logits.append(scores)
        return dict(row=row, logits=logits, hidden=hiddens)

    results = {name: run(row, name == "base") for name, row in cases.items()}
    b = results["base"]["logits"]
    for name in ("sibling_token", "sibling_length"):
        assert results[name]["logits"][1] == b[1]
        assert results[name]["logits"][0][0] == b[0][0]
        assert results[name]["logits"][0][1] != b[0][1]
    assert results["other_question"]["logits"][0] == b[0]
    assert results["question_length"]["logits"][1] == b[1]
    assert results["candidate_order"]["logits"] == [list(reversed(x)) for x in b]
    assert results["question_order"]["logits"] == list(reversed(b))
    dest.write_text(json.dumps(dict(schema=1, policy="f32-branch-local-positions-full-tree-causal-linear",
                                    source_revision=REV, modeling_sha256=MODEL_SHA, config_sha256=CONFIG_SHA,
                                    weights_sha256=WEIGHTS_SHA, weights_size=WEIGHTS_SIZE,
                                    transformers_version=transformers.__version__, torch_version=torch.__version__,
                                    transformers_qwen_sha256=TRANSFORMERS_SHA, cases=results),
                               separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
