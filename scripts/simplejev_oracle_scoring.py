#!/usr/bin/env python3
"""Capture model-free scoring observations from a pinned external Simple-JEV checkout.

Run with a clean featherless-ai/simple-jev checkout at
b02aa81c915a8193759b3cd33fef74721d6e005b, using Python with Pydantic 2
and NumPy. No weights, HF adapter, service, or prompt strings are exported.
"""

import hashlib
import json
import pathlib
import subprocess
import sys

REVISION = "b02aa81c915a8193759b3cd33fef74721d6e005b"
SCORING_SHA = "b1b1303ace5218fb40904e7aaf5a1ad01aa3d072c30678d45e921bb0e8613dfa"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: simplejev_oracle_scoring.py UPSTREAM_CHECKOUT OUTPUT.json")
    root, destination = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"wrong upstream revision: {revision}")
    source = root / "common" / "response_scoring.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SCORING_SHA:
        raise SystemExit("upstream scorer changed")
    sys.path.insert(0, str(root))
    from common import prepare_prompt, build_answers  # imported only from the checked checkout

    # Independent, synthetic request text; the upstream template stays upstream.
    request = {
        "model": "fixture", "state": "A synthetic state.",
        "questions": {
            "pick": {"type": "choice", "instructions": "Choose one.", "criteria": {"alpha": None, "beta": None, "gamma": None}},
            "rating": {"type": "score", "instructions": "Rate it.", "criteria": ["r0", "r1", "r2", "r3"]},
            "binary": {"type": "noul", "instructions": "Estimate it."},
        },
    }
    plan = prepare_prompt(request)
    branches = [(q.branch_id, q.question_id, list(q.output_labels)) for q in plan.questions]
    if branches != [("0", "pick", ["A", "B", "C"]), ("1", "rating", ["0", "1", "2", "3"]), ("2", "binary", list("123456789"))]:
        raise SystemExit(f"unexpected upstream branch mapping: {branches}")

    results = []
    cases = [
        ("unequal", [[0, 2, -1], [-1, 1, 2, 3], list(range(9))]),
        ("tie", [[3, 3, -2], [0, 0, 0, 0], [0] * 9]),
        ("extreme", [[-1000, 1000, 0], [1000, -1000, 0, -1000], [-1000] * 8 + [1000]]),
    ]
    for name, rows in cases:
        inputs = {branch: {label: float(value) for label, value in zip(labels, row)}
                  for (branch, _, labels), row in zip(branches, rows)}
        answer = build_answers(plan, inputs)
        results.append({
            "name": name,
            "logits": rows,
            "choice": {"id": answer["pick"]["choice"], "probabilities": list(answer["pick"]["probabilities"].values())},
            "score": {"value": answer["rating"]["score"], "probabilities": list(answer["rating"]["probabilities"].values())},
            "noul": {"value": answer["binary"]["noul"]},
        })

    # The oracle must reject incomplete labels, missing/extra branches and nonfinite rows.
    good = {q.branch_id: {label: 0.0 for label in q.output_labels} for q in plan.questions}
    rejected = {}
    for name, modifier in (
        ("missing_label", lambda x: x["0"].pop("A")),
        ("extra_branch", lambda x: x.update({"extra": {"A": 0.0}})),
        ("nonfinite", lambda x: x["1"].update({"0": float("nan")})),
    ):
        inputs = {branch: row.copy() for branch, row in good.items()}
        modifier(inputs)
        try:
            build_answers(plan, inputs)
        except (ValueError, TypeError, KeyError) as exc:
            rejected[name] = type(exc).__name__
        else:
            raise SystemExit(f"upstream accepted {name}")

    fixture = {"schema": 1, "oracle_repo": "featherless-ai/simple-jev", "oracle_revision": REVISION,
               "scoring_sha256": SCORING_SHA, "branches": [dict(branch=branch, question=question, labels=labels) for branch, question, labels in branches],
               "cases": results, "rejected": rejected}
    destination.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
