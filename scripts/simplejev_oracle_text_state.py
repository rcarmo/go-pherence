#!/usr/bin/env python3
"""Capture a small text-state validation subset from pinned Simple-JEV Python.

No model, prompt renderer, or HTTP server is imported. Run with Pydantic 2.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "b02aa81c915a8193759b3cd33fef74721d6e005b"
SCHEMA_SHA = "6fa1c1215e8fc9de7aedbee77db6520bbb811922df9c45858bf78c4e4a02ceac"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: simplejev_oracle_text_state.py UPSTREAM_CHECKOUT OUTPUT.json")
    upstream, output = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip() != REV:
        raise SystemExit("wrong upstream revision")
    if hashlib.sha256((upstream / "common/request_schema.py").read_bytes()).hexdigest() != SCHEMA_SHA:
        raise SystemExit("upstream schema changed")
    sys.path.insert(0, str(upstream))
    from common import ClassifierRequest

    def request(state=""):  # synthetic input, not an upstream prompt
        return {"model": "fixture", "state": state, "questions": {
            "pick": {"type": "choice", "instructions": "", "criteria": {"first": None, "second": "text"}},
            "level": {"type": "score", "instructions": None, "criteria": ["low", None, "high"]},
            "truth": {"type": "noul", "instructions": "Is it true?", "criteria": {"false": None}},
        }, "options": {"raw_logits": True}}

    cases = []
    for name, value, accepted in [
        ("empty_state_and_null", request(), True),
        ("ordinary_text", request("plain text"), True),
        ("unknown_top", dict(request(), unrelated=7), True),
        ("both_contexts", dict(request(), messages=[{"role": "user", "content": "hi"}]), False),
        ("no_context", {k: v for k, v in request().items() if k != "state"}, False),
        ("unknown_question", {**request(), "questions": {"pick": {**request()["questions"]["pick"], "unknown": 1}}}, False),
        ("unknown_options", {**request(), "options": {"raw_logits": True, "unknown": True}}, False),
        ("one_choice", {**request(), "questions": {"pick": {"type": "choice", "instructions": "", "criteria": {"one": None}}}}, False),
        ("one_score", {**request(), "questions": {"score": {"type": "score", "instructions": "", "criteria": ["one"]}}}, False),
        ("empty_questions", {**request(), "questions": {}}, False),
        ("noul_wrong_key", {**request(), "questions": {"truth": {"type": "noul", "instructions": "", "criteria": {"perhaps": None}}}}, False),
        ("bad_type", {**request(), "questions": {"pick": {"type": "unknown", "instructions": "", "criteria": {"one": None, "two": None}}}}, False),
    ]:
        try:
            got = ClassifierRequest.model_validate(value)
        except Exception as error:
            if accepted:
                raise SystemExit(f"upstream rejected {name}: {error}") from error
            cases.append({"name": name, "request": value, "accepted": False})
            continue
        if not accepted:
            raise SystemExit(f"upstream accepted {name}")
        cases.append({"name": name, "request": value, "accepted": True,
                      "model": got.model, "state": got.state, "raw_logits": got.options.raw_logits,
                      "questions": [{"id": key, "type": q.type,
                                     "instructions": q.instructions,
                                     "criteria": q.criteria}
                                    for key, q in got.questions.items()]})
    output.write_text(json.dumps({"schema": 1, "oracle_repo": "featherless-ai/simple-jev",
                                  "oracle_revision": REV, "request_schema_sha256": SCHEMA_SHA,
                                  "cases": cases}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
