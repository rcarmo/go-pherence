#!/usr/bin/env python3
"""Pin structured-state canonical JSON observations without prompt text.

Apache upstream source stays in the read-only oracle checkout. The output
contains synthetic input and canonical JSON only; no prompt or model output.
"""
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
        raise SystemExit("usage: simplejev_oracle_structured_state.py UPSTREAM_CHECKOUT OUTPUT.json")
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

    samples = (
        '{"z":2,"a":{"b":true,"a":[null,"café",0]}}',
        '[{"z":"last","a":"first"},[],false]',
        '{"line":"a\\nb","combining":"e\\u0301","emoji":"😀"}',
        '{}', '[]', '"plain state"',
    )
    accepted = []
    for raw in samples:
        state = json.loads(raw)
        req = ClassifierRequest.model_validate(dict(model="fixture", state=state,
                                                    questions={"q": dict(type="noul", instructions="") }))
        accepted.append(dict(input=raw, canonical=canonical(req.state), kind=type(req.state).__name__))
    rejected = []
    for name, raw in (("nonfinite", '{"n":NaN}'), ("state_number", '42'), ("state_boolean", 'true'),
                      ("no_state", 'null')):
        try:
            state = json.loads(raw)
            req = ClassifierRequest.model_validate(dict(model="fixture", state=state,
                                                        questions={"q": dict(type="noul", instructions="") }))
            canonical(req.state)
        except (ValueError, TypeError) as err:
            rejected.append(dict(name=name, input=raw, error_type=type(err).__name__))
        else:
            raise SystemExit(f"unexpectedly accepted {name}")
    dest.write_text(json.dumps(dict(schema=1, revision=REV, request_schema_sha256=SCHEMA_SHA,
                                    prompt_builder_sha256=BUILDER_SHA,
                                    accepted=accepted, rejected=rejected),
                               ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
