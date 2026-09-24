#!/usr/bin/env python3
"""Pin MoJev text prompt and packed-token outputs with the released tokenizer.

Runs the MIT collator from pinned source; does not load model weights or images.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
SCHEMA_SHA = "63562314b7e6c1ed7e7629497ccc0022771e876fac7730f5ce7d038b50be0df7"
FULL_SHA = "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79"
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
CONFIG_SHA = "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: mojev_oracle_released_packing.py PINNED_SOURCE TOKENIZER_DIR OUTPUT.json")
    source, checkpoint, dest = map(lambda x: pathlib.Path(x).resolve(), sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("wrong source revision")
    for path, expected in ((source / "mojev/schema.py", SCHEMA_SHA), (source / "mojev/full.py", FULL_SHA),
                           (checkpoint / "tokenizer.json", TOKENIZER_SHA), (checkpoint / "tokenizer_config.json", CONFIG_SHA)):
        if digest(path) != expected:
            raise SystemExit(f"asset hash mismatch: {path}")
    sys.path.insert(0, str(source))
    from transformers import AutoTokenizer
    from mojev.schema import Field, Schema
    from mojev.data import validate
    from mojev.full import packed_collate

    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, use_fast=False)
    schema = Schema((Field("next_step", "choice", ("approve", "reject"), "Decide now?"),
                     Field("risk", "choice", ("low", "high"), "Assess risk")))
    rows = [dict(context="state café — urgent 42", labels={"next_step": "approve", "risk": "low"},
                 options={"risk": ["low", "very high"]}),
            dict(context="short", labels={"next_step": "reject", "risk": "high"})]
    examples = [validate(row, schema) for row in rows]
    max_state, max_question = 12, 24
    batch = packed_collate(tok, schema, max_state, field_tokens_max=max_question)(examples)
    keys = ("packed_ids", "packed_mask", "context_span", "field_span", "option_span", "option_mask")
    result = dict(schema=1, source_revision=REV, schema_sha256=SCHEMA_SHA, full_sha256=FULL_SHA,
                  tokenizer_sha256=TOKENIZER_SHA, tokenizer_config_sha256=CONFIG_SHA,
                  state_limit=max_state, question_limit=max_question, pad_id=tok.pad_token_id,
                  fields=[dict(name=f.name, kind=f.kind, options=list(f.options), description=f.description) for f in schema],
                  rows=[dict(context=row["context"], menus=[row.get("options", {}).get(f.name, list(f.options)) for f in schema]) for row in rows],
                  prompts=list(schema.prompts), expected={key: batch[key].int().tolist() for key in keys})
    dest.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
