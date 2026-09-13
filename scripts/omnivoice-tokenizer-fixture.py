#!/usr/bin/env python3
"""Generate a tiny OmniVoice tokenizer fixture for Go tests.

It preserves the real OmniVoice pre-tokenizer regex and the exact upstream token
IDs needed by a small parity corpus without copying the full 700k-line
`tokenizer.json` into this repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

CASES = [
    "can't",
    "we'll I'm",
    "1 23 456",
    "foo\t bar",
    "a  b",
    "a\t\tb",
    "\n ",
    "<|denoise|><|lang_start|>en<|lang_end|><|instruct_start|>None<|instruct_end|>",
    "<|text_start|>can't 123\n[laughter]<|text_end|>",
]
SPECIALS = {
    "<|denoise|>",
    "<|lang_start|>",
    "<|lang_end|>",
    "<|instruct_start|>",
    "<|instruct_end|>",
    "<|text_start|>",
    "<|text_end|>",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", default=Path("testdata/omnivoice/tokenizer.json"), type=Path)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer_json = json.loads((args.model / "tokenizer.json").read_text())
    split_regex = tokenizer_json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]

    vocab = {}
    for text in CASES:
        enc = tokenizer(text, add_special_tokens=False)
        for token, token_id in zip(tokenizer.convert_ids_to_tokens(enc.input_ids), enc.input_ids):
            if token not in SPECIALS:
                vocab[token] = token_id

    added_tokens = []
    for entry in tokenizer_json.get("added_tokens", []):
        if entry["content"] in SPECIALS:
            added_tokens.append(
                {
                    "id": entry["id"],
                    "content": entry["content"],
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": bool(entry.get("special", False)),
                }
            )

    fixture = {
        "version": "1.0",
        "cases": [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)} for text in CASES],
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": split_regex},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": dict(sorted(vocab.items(), key=lambda kv: kv[1])),
            "merges": None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
