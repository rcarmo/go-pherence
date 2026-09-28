#!/usr/bin/env python3
"""Pin offline, released MoJev tokenizer IDs without loading model weights.

Read only the approved tokenizer.json and tokenizer_config.json. Reject a
checkpoint directory whose tokenizer files differ from the pinned revision.
"""
import hashlib
import json
import pathlib
import sys

TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
CONFIG_SHA = "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: mojev_oracle_tokenizer.py CHECKPOINT_DIR OUTPUT.json")
    checkpoint, output = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:])
    if digest(checkpoint / "tokenizer.json") != TOKENIZER_SHA or digest(checkpoint / "tokenizer_config.json") != CONFIG_SHA:
        raise SystemExit("tokenizer asset hash mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, use_fast=False)
    samples = (
        "state", "A", "B", "é", "Hello world", "42", "a  b", "  hi  ", "\nq?",
        "<|im_start|>", "<|im_end|>", "<|image_pad|>",
        "café e\u0301", "東京 7", "€😀", "line\n\nend", "can't — won't", "\t x\t",
    )
    cases = [dict(text=text, ids=tokenizer(text, truncation=False)["input_ids"]) for text in samples]
    result = dict(schema=1, model_revision="0c8695b6252f4205907433d4e196a94f032e60c3",
                  tokenizer_sha256=TOKENIZER_SHA, tokenizer_config_sha256=CONFIG_SHA,
                  tokenizer_class=type(tokenizer).__name__, pad_id=tokenizer.pad_token_id,
                  eos_id=tokenizer.eos_token_id, cases=cases)
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
