#!/usr/bin/env python3
"""Record structural v1 prompt/label boundaries, excluding Apache prompt text.

Runs the pinned Apache upstream HF compiler with its byte tokenizer test double.
Outputs only SHA-256/lengths for rendered prompts and branch token paths,
plus public branch IDs, model-facing labels, output token IDs and rejections.
No prompt or upstream source text is copied into the MIT Go package.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "b02aa81c915a8193759b3cd33fef74721d6e005b"
PINS = {
    "hf-server/hf_server.py": "a96440b203d047800498479f85ecb417da340b77e837a5dc102fcb81938e3b1b",
    "hf-server/tests/conftest.py": "cafc09af4e292c1ed6fb797fdf725dadda5b692db9926fcc334ead4bb45d3611",
    "common/prompt_builder.py": "14a885d3bfa44b0c8aa0ffc9ce84611a459957d938e1847f96d93c7f1840fa3d",
    "common/request_schema.py": "6fa1c1215e8fc9de7aedbee77db6520bbb811922df9c45858bf78c4e4a02ceac",
}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: simplejev_oracle_prompt_boundary.py UPSTREAM_CHECKOUT OUTPUT.json")
    source, dest = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    for file, expected in PINS.items():
        if hashlib.sha256((source / file).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"source hash mismatch: {file}")
    sys.path[:0] = [str(source / "hf-server/tests"), str(source / "hf-server"), str(source)]
    from hf_server import PromptCompiler
    from conftest import Tokenizer

    request = {
        "model": "m", "state": "red", "questions": {
            "color": {"type": "choice", "instructions": "Color?", "criteria": {"red": None, "blue": None}},
            "level": {"type": "score", "instructions": "Level?", "criteria": list(map(str, range(11)))},
            "truth": {"type": "noul", "instructions": "Red?"},
        },
    }
    compiled = PromptCompiler(Tokenizer()).compile(request)
    branches = []
    for branch, question in zip(compiled.branches, compiled.plan.questions):
        branches.append(dict(id=branch.branch_id,
                             token_count=len(branch.token_ids),
                             token_sha256=hashlib.sha256(bytes(branch.token_ids)).hexdigest(),
                             answer_prefix_bytes=len(branch.answer_prefix.encode("utf-8")),
                             answer_prefix_sha256=hashlib.sha256(branch.answer_prefix.encode("utf-8")).hexdigest(),
                             labels=list(question.output_labels),
                             label_token_ids=branch.output_ids,
                             messages=len(branch.messages)))
    # Use the same synthetic failure mechanisms as upstream tests, but keep
    # only error categories. No prompt text is written to the fixture.
    class Unstable(Tokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            return ids[:-2] + [0] if text.endswith('"A') else ids
    class Duplicate(Tokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            return ids[:-1] + [ord("A")] if text.endswith('"B') else ids
    failures = []
    for name, tokenizer in (("retokenized_boundary", Unstable()), ("duplicate_label_id", Duplicate())):
        try:
            PromptCompiler(tokenizer).compile(request)
        except ValueError as err:
            failures.append(dict(name=name, category="single-token" if "single-token" in str(err) else "distinct" if "distinct" in str(err) else "other"))
        else:
            raise SystemExit(f"upstream accepted {name}")
    dest.write_text(json.dumps(dict(schema=1, source_revision=REV, source_sha256=PINS,
                                    tokenizer="upstream byte test double", branches=branches,
                                    failures=failures), separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
