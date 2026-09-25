#!/usr/bin/env python3
"""Pin text-only MoJev option rendering and validation observations.

The cases exercise MIT serving helpers without loading model weights or HTTP.
Structured/media state and full SDK validation are outside this boundary.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
SERVE_SHA = "7d244785d11eb5cdb490080f24f1307b2c5df8126a6b0858c4f1db341d838c8d"
FULL_SHA = "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: mojev_oracle_text_request.py PINNED_SOURCE OUTPUT.json")
    source, dest = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    for path, digest in ((source / "mojev/serve.py", SERVE_SHA), (source / "mojev/full.py", FULL_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit("source hash mismatch")
    sys.path.insert(0, str(source))
    from mojev.serve import option_texts, _render, ValidationFailure
    from mojev.full import sort_candidates

    cases = []
    for name, payload in (
        ("mixed", dict(model="mojev-latest", state="State é", questions={
            "decision": dict(type="choice", instructions="Pick now", criteria={"z": "last", "a": "first", "m": None}),
            "binary": dict(type="noul", criteria={"false": "not yet", "true": "ready"}),
            "rating": dict(type="score", criteria=["low", "medium", "high"]),
        })),
        ("noul_defaults", dict(model="mojev-latest", state="x", questions={"ok": dict(type="noul")})),
        ("choice_empty_description", dict(model="mojev-latest", state="x", questions={
            "act": dict(type="choice", criteria={"yes": "", "no": None})})),
    ):
        fields = []
        for key, question in payload["questions"].items():
            options, kind = option_texts(key, question)
            order = sort_candidates(options)
            fields.append(dict(id=key, kind=kind, keys=(["false", "true"] if kind == "noul" else list(question.get("criteria", {}).keys()) if kind == "choice" else [str(i) for i in range(len(options))]),
                               options=options, sorted_indices=order,
                               sorted_options=[options[i] for i in order],
                               instructions=_render(question["instructions"]) if question.get("instructions") is not None else ""))
        cases.append(dict(name=name, request=payload, state_text=_render(payload["state"]), fields=fields))
    invalid = []
    for name, question in (
        ("missing_type", {}), ("bad_type", {"type": "unknown"}),
        ("choice_empty", {"type": "choice", "criteria": {}}),
        ("noul_wrong_criteria", {"type": "noul", "criteria": ["unexpected"]}),
        ("score_empty", {"type": "score", "criteria": []}),
    ):
        try:
            option_texts("q", question)
        except ValidationFailure as error:
            invalid.append(dict(name=name, question=question, detail=error.detail))
        else:
            raise SystemExit(f"upstream unexpectedly accepted {name}")
    result = dict(schema=1, source_revision=REV, serve_sha256=SERVE_SHA,
                  full_sha256=FULL_SHA, cases=cases, invalid=invalid)
    dest.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
