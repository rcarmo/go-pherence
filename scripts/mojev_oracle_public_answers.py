#!/usr/bin/env python3
"""Pin synthetic public-answer assembly from the MIT MoJev serving helpers.

Rows are injected logits; this does not execute the released encoder or scorer.
"""
import hashlib
import json
import math
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
SERVE_SHA = "7d244785d11eb5cdb490080f24f1307b2c5df8126a6b0858c4f1db341d838c8d"
FULL_SHA = "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: mojev_oracle_public_answers.py PINNED_SOURCE OUTPUT.json")
    source, output = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    for path, digest in ((source / "mojev/serve.py", SERVE_SHA), (source / "mojev/full.py", FULL_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit("source hash mismatch")
    sys.path.insert(0, str(source))
    from mojev.serve import build_answer
    from mojev.full import sort_candidates, unsort

    def softmax(row):
        peak = max(row)
        exp = [math.exp(value - peak) for value in row]
        return [value / sum(exp) for value in exp]

    cases = []
    for name, kind, criteria, logits, instructions in (
        ("choice_unsort", "choice", {"z": "last", "a": "first", "m": None}, [4.0, -1.0, 4.0], "Choose"),
        ("choice_tie", "choice", {"z": "last", "a": "first"}, [2.0, 2.0], None),
        ("noul", "noul", {"false": "no", "true": "yes"}, [0.25, -0.5], None),
        ("score", "score", ["low", "mid", "high"], [-2.0, 0.5, 1.0], "Assess"),
        ("score_extreme", "score", ["cold", "warm"], [-300.0, 300.0], None),
    ):
        question = {"type": kind, "criteria": criteria}
        if instructions is not None:
            question["instructions"] = instructions
        if kind == "choice":
            keys = list(criteria.keys())
            options = [key if value in (None, "") else f"{key}: {value}" for key, value in criteria.items()]
        elif kind == "noul":
            keys = ["false", "true"]
            options = [criteria["false"], criteria["true"]]
        else:
            keys = [str(i) for i in range(len(criteria))]
            options = criteria
        order = sort_candidates(options)
        probs = unsort(order, softmax(logits))
        answer = build_answer(name, question, kind, probs)
        cases.append(dict(name=name, kind=kind, keys=keys, options=options,
                          sorted_indices=order, sorted_logits=logits,
                          probabilities=probs, answer=answer))
    result = dict(schema=1, source_revision=REV, serve_sha256=SERVE_SHA, full_sha256=FULL_SHA, cases=cases)
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
