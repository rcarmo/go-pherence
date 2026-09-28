#!/usr/bin/env python3
"""Pin small MoJev tree-mask observations; no checkpoint is loaded.

Run from the pinned MIT source checkout with CPU torch and transformers installed.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
MODEL_SHA = "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: mojev_oracle_tree_mask.py PINNED_SOURCE OUTPUT.json")
    source, dest = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("wrong source revision")
    if hashlib.sha256((source / "mojev/modeling.py").read_bytes()).hexdigest() != MODEL_SHA:
        raise SystemExit("source model hash mismatch")
    sys.path.insert(0, str(source))
    import torch
    from mojev.modeling import PackedScorer

    cases = []
    for name, length, options, truncate_state in (("two_questions", 12, 2, False), ("padding_and_empty_state_tail", 16, 3, True)):
        fields = 2
        state = [1] * (length - fields * (1 + options)) + [0] * (fields * (1 + options))
        question = [[0] * length for _ in range(fields)]
        candidate = [[[0] * length for _ in range(options)] for _ in range(fields)]
        cursor = length - fields * (1 + options)
        for f in range(fields):
            question[f][cursor] = 1
            cursor += 1
            for n in range(options):
                candidate[f][n][cursor] = 1
                cursor += 1
        if truncate_state:
            state[4:] = [0] * (length - 4)
        mask = PackedScorer.build_mask(None, torch.tensor([state], dtype=torch.float32),
                                        torch.tensor([question], dtype=torch.float32),
                                        torch.tensor([candidate], dtype=torch.float32))
        allowed = (mask[0, 0] == 0.0).int().tolist()
        if mask.shape != (1, 1, length, length) or not all(any(row) for row in allowed):
            raise SystemExit("unexpected upstream mask geometry")
        cases.append(dict(name=name, state=state, questions=question, candidates=candidate, allowed=allowed))
    result = dict(schema=1, source_repo="MoLeMo-Lab/mojev", source_revision=REV,
                  modeling_sha256=MODEL_SHA, cases=cases)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
