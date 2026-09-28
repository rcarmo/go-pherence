#!/usr/bin/env python3
"""Capture public, non-advanced model-free upstream response observations.

Requires pinned featherless-ai/simple-jev and Pydantic 2/NumPy 2. No weights,
HF adapter, prompt strings or model runtime are exported to the fixture.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "b02aa81c915a8193759b3cd33fef74721d6e005b"
SCORER_SHA = "b1b1303ace5218fb40904e7aaf5a1ad01aa3d072c30678d45e921bb0e8613dfa"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: simplejev_oracle_public_response.py UPSTREAM_CHECKOUT OUTPUT.json")
    upstream, output = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip() != REV:
        raise SystemExit("wrong upstream revision")
    if hashlib.sha256((upstream / "common/response_scoring.py").read_bytes()).hexdigest() != SCORER_SHA:
        raise SystemExit("upstream scorer changed")
    sys.path.insert(0, str(upstream))
    from common import prepare_prompt, build_response

    request = {"model": "fixture", "state": "synthetic text", "questions": {
        "pick": {"type": "choice", "instructions": "Select", "criteria": {"alpha": None, "beta": "second", "gamma": "third"}},
        "rating": {"type": "score", "instructions": "Rank", "criteria": ["low", "mid", "high"]},
        "truth": {"type": "noul", "instructions": "Is it true?"},
    }}
    plan = prepare_prompt(request)
    branches = [(q.branch_id, q.question_id, list(q.output_labels)) for q in plan.questions]
    if branches != [("0", "pick", ["A", "B", "C"]), ("1", "rating", ["0", "1", "2"]), ("2", "truth", list("123456789"))]:
        raise SystemExit("unexpected branch mapping")

    cases = []
    for name, rows in (
        ("tie", [[2, 2, -1], [-1, 1, 2], list(range(9))]),
        ("extreme", [[1000, -1000, 0], [0, 1000, -1000], [-1000] * 8 + [1000]]),
    ):
        logits = {q.branch_id: {label: float(value) for label, value in zip(q.output_labels, rows[i])}
                  for i, q in enumerate(plan.questions)}
        response = build_response(plan, logits, input_tokens=12, output_tokens=0)
        cases.append({"name": name, "logits": rows, "response": response})

    fixture = {"schema": 1, "oracle_repo": "featherless-ai/simple-jev", "oracle_revision": REV,
               "scorer_sha256": SCORER_SHA, "request": request,
               "branches": [dict(branch=branch, question=question, labels=labels) for branch, question, labels in branches],
               "cases": cases}
    output.write_text(json.dumps(fixture, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
