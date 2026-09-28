#!/usr/bin/env python3
"""Pin multi-question MoJev public response from injected sorted logits.

The approved tokenizer supplies usage counts. No checkpoint weights or
encoder are loaded; upstream helper functions supply sorting and answers.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
SERVE_SHA = "7d244785d11eb5cdb490080f24f1307b2c5df8126a6b0858c4f1db341d838c8d"
FULL_SHA = "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
CONFIG_SHA = "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: mojev_oracle_text_response.py PINNED_SOURCE TOKENIZER_DIR OUTPUT.json")
    source, tokenizer_dir, dest = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    for path, sha in ((source / "mojev/serve.py", SERVE_SHA), (source / "mojev/full.py", FULL_SHA),
                      (tokenizer_dir / "tokenizer.json", TOKENIZER_SHA),
                      (tokenizer_dir / "tokenizer_config.json", CONFIG_SHA)):
        if digest(path) != sha:
            raise SystemExit(f"hash mismatch: {path}")
    sys.path.insert(0, str(source))
    import torch
    from transformers import AutoTokenizer
    from mojev.serve import build_answer, option_texts
    from mojev.full import sort_candidates, unsort, packed_collate
    from mojev.schema import Field, Schema
    from mojev.data import Example
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False, use_fast=False)
    request = dict(model="mojev-latest", state="State é", questions={
        "action": dict(type="choice", instructions="Pick now", criteria={"z": "last", "a": "first", "m": None}),
        "binary": dict(type="noul", criteria={"false": "not yet", "true": "ready"}),
        "risk": dict(type="score", criteria=["low", "medium", "high"]),
    })
    names, kinds, orders, fields, menus = [], [], [], [], []
    for name, question in request["questions"].items():
        options, kind = option_texts(name, question)
        order = sort_candidates(options)
        names.append(name); kinds.append(kind); orders.append(order)
        menu = tuple(options[i] for i in order)
        menus.append(menu)
        fields.append(Field(name, "choice", menu, question.get("instructions", "")))
    schema = Schema(tuple(fields))
    collated = packed_collate(tokenizer, schema, context_tokens=128)([Example(request["state"], tuple(0 for _ in names), tuple(menus))])
    # Sorted logit rows, injected in place of the model forward pass.
    logits = [[2.0, -0.5, 2.0], [0.25, -0.5], [-2.0, 0.5, 1.0]]
    answers = {}
    for i, name in enumerate(names):
        row = torch.tensor(logits[i], dtype=torch.float32)
        probs = unsort(orders[i], row.softmax(-1).tolist())
        answers[name] = build_answer(name, request["questions"][name], kinds[i], probs)
    result = dict(schema=1, source_revision=REV, serve_sha256=SERVE_SHA, full_sha256=FULL_SHA,
                  tokenizer_sha256=TOKENIZER_SHA, tokenizer_config_sha256=CONFIG_SHA,
                  request=request, sorted_logits=logits,
                  input_tokens=int(collated["packed_mask"].sum()),
                  response=dict(model=request["model"], answers=answers,
                                usage=dict(input_tokens=int(collated["packed_mask"].sum()), output_tokens=0)))
    dest.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
