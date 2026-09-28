#!/usr/bin/env python3
"""Record bounded structured-state request observations, without prompts/models."""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "b02aa81c915a8193759b3cd33fef74721d6e005b"
SCHEMA_SHA = "6fa1c1215e8fc9de7aedbee77db6520bbb811922df9c45858bf78c4e4a02ceac"
BUILDER_SHA = "14a885d3bfa44b0c8aa0ffc9ce84611a459957d938e1847f96d93c7f1840fa3d"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: simplejev_oracle_structured_request.py UPSTREAM_CHECKOUT OUTPUT.json")
    source, dest = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    for path, expected in ((source / "common/request_schema.py", SCHEMA_SHA),
                           (source / "common/prompt_builder.py", BUILDER_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"source hash mismatch: {path}")
    sys.path.insert(0, str(source))
    from common.request_schema import ClassifierRequest
    from common.prompt_builder import canonical

    questions = {
        "pick": {"type": "choice", "instructions": "select", "criteria": {"first": None, "second": "text"}},
        "level": {"type": "score", "instructions": None, "criteria": ["low", None, "high"]},
        "truth": {"type": "noul", "instructions": "Is it true?", "criteria": {"false": None}},
    }
    cases = []
    for name, state in (
        ("object", {"z": 2, "a": {"b": True, "a": [None, "café", 0]}}),
        ("array", [{"z": "last", "a": "first"}, [], False]),
        ("empty_object", {}), ("empty_array", []),
    ):
        value = {"model": "fixture", "state": state, "questions": questions,
                 "options": {"raw_logits": False}, "ignored_top": 7}
        req = ClassifierRequest.model_validate(value)
        cases.append({"name": name, "request": value, "accepted": True,
                      "model": req.model, "canonical_state": canonical(req.state),
                      "raw_logits": req.options.raw_logits,
                      "question_ids": list(req.questions)})
    for name, value in (
        ("no_state", {"model": "fixture", "questions": questions}),
        ("null_state", {"model": "fixture", "state": None, "questions": questions}),
        ("number_state", {"model": "fixture", "state": 42, "questions": questions}),
        ("boolean_state", {"model": "fixture", "state": True, "questions": questions}),
        ("both_contexts", {"model": "fixture", "state": {}, "messages": [{"role": "user", "content": "hi"}], "questions": questions}),
    ):
        try:
            ClassifierRequest.model_validate(value)
        except ValueError:
            cases.append({"name": name, "request": value, "accepted": False})
        else:
            raise SystemExit(f"upstream accepted {name}")
    dest.write_text(json.dumps({"schema": 1, "revision": REV, "request_schema_sha256": SCHEMA_SHA,
                                "prompt_builder_sha256": BUILDER_SHA, "cases": cases},
                               ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
