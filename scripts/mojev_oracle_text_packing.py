#!/usr/bin/env python3
"""Pin text-only MoJev packing observations from MIT upstream, no model weights.

The small tokenizer maps each UTF-8 byte to byte+1, as upstream's model-free
ByteTokenizer does. It is not a released Qwen3.5 tokenizer parity fixture.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
FULL_SHA = "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: mojev_oracle_text_packing.py PINNED_SOURCE OUTPUT.json")
    source, dest = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    if hashlib.sha256((source / "mojev/full.py").read_bytes()).hexdigest() != FULL_SHA:
        raise SystemExit("source packing hash mismatch")
    sys.path.insert(0, str(source))
    from mojev.schema import Field, Schema
    from mojev.data import validate
    from mojev.full import packed_collate

    class ByteTokenizer:
        pad_token_id = 0
        def __call__(self, texts, truncation=False, max_length=None, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            rows = []
            for text in texts:
                ids = [b + 1 for b in text.encode("utf-8")]
                rows.append((ids[:max_length] if max_length else ids) or [1])
            return {"input_ids": rows}

    tokenizer = ByteTokenizer()
    schema = Schema((Field("q0", "choice", ("A", "B"), "First?"),
                     Field("q1", "choice", ("x", "y"), "Second?")))
    rows = [dict(context="state-é", labels={"q0": "A", "q1": "x"}, options={"q1": ["x", "long candidate"]}),
            dict(context="s", labels={"q0": "B", "q1": "x"})]
    # The first row's override exceeds the other row only in token length;
    # it tests batch padding without exceeding schema cardinality.
    examples = [validate(row, schema) for row in rows]
    batch = packed_collate(tokenizer, schema, context_tokens=5, field_tokens_max=4)(examples)
    material = []
    prompts = list(schema.prompts)
    for row in rows:
        menus = [row.get("options", {}).get(f.name, list(f.options)) for f in schema]
        material.append(dict(state=tokenizer(row["context"], max_length=5)["input_ids"][0],
                             questions=[tokenizer(p, max_length=4)["input_ids"][0] for p in prompts],
                             candidates=[[tokenizer(text)["input_ids"][0] for text in menu] for menu in menus]))
    keys = ("packed_ids", "packed_mask", "context_span", "field_span", "option_span", "option_mask")
    result = dict(schema=1, source_revision=REV, full_sha256=FULL_SHA, pad_id=0,
                  input=material, expected={key: batch[key].int().tolist() for key in keys})
    dest.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
