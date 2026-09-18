#!/usr/bin/env python3
"""Generate a reduced OmniVoice tokenizer fixture for Go parity tests.

The fixture preserves the real OmniVoice NFC normalizer, byte-level split regex,
selected special tokens, and the minimal vocab/merge closure needed for the test
corpus. This keeps the repository fixture small while still exercising the real
multilingual tokenizer behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

CASES = [
    "can't",
    "we'll I'm you're they've",
    "café",
    "cafe\u0301",
    "Olá, ação! São Tomé e Príncipe.",
    "你好世界 mixed Latin 123",
    "日本語テスト ひらがな カタカナ 漢字",
    "한국어 테스트 한글 123",
    "مرحبا بالعالم العَرَبِيَّة",
    "שלום עולם עִבְרִית",
    "नमस्ते दुनिया हिन्दी १२३",
    "สวัสดีโลก ไทย ๑๒๓",
    "👩‍💻 ❤️ 👨‍👩‍👧‍👦 a⃝",
    "١2३ １２3 abc१२٣def",
    "foo\t bar",
    "a  b",
    "a\t\tb",
    "\n ",
    "foo\n\nbar",
    "a\u00a0b a\u2003b a\u2028b a\u2009b a\u3000b",
    "a\x0bb a\x0cb a\u200db a\ufeffb",
    "line1\r\nline2\rline3\u0085line4",
    "<|denoise|><|lang_start|>pt<|lang_end|><|instruct_start|>[laughter]<|instruct_end|>",
    "<|text_start|>[breath] uh… <<nonverbal>> hello<|text_end|>",
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
MODEL_FIELDS = (
    "type",
    "dropout",
    "unk_token",
    "continuing_subword_prefix",
    "end_of_word_suffix",
    "fuse_unk",
    "byte_fallback",
    "ignore_merges",
)


def load_merges(model_json: dict) -> list[tuple[str, str]]:
    merges = model_json.get("merges") or []
    if not merges:
        return []
    if isinstance(merges[0], str):
        return [tuple(merge.split(" ", 1)) for merge in merges]
    return [tuple(merge) for merge in merges]


def build_merge_closure(tokenizer_json: dict, tokenizer: AutoTokenizer) -> tuple[dict[str, int], list[list[str]]]:
    model_json = tokenizer_json["model"]
    vocab = model_json["vocab"]
    merges = load_merges(model_json)
    merge_by_output = {}
    for rank, (left, right) in enumerate(merges):
        merge_by_output.setdefault(left + right, (left, right, rank))

    selected: set[str] = set()

    def add_token(token: str) -> None:
        if token in selected:
            return
        selected.add(token)
        pair = merge_by_output.get(token)
        if pair is None:
            return
        left, right, _ = pair
        add_token(left)
        add_token(right)

    for text in CASES:
        encoded = tokenizer(text, add_special_tokens=False)
        for token in tokenizer.convert_ids_to_tokens(encoded.input_ids):
            if token not in SPECIALS:
                add_token(token)

    reduced_vocab = dict(sorted(((token, vocab[token]) for token in selected), key=lambda item: item[1]))
    reduced_merges = [
        [left, right]
        for left, right, rank in sorted(
            (pair for output, pair in merge_by_output.items() if output in selected and pair[0] in selected and pair[1] in selected),
            key=lambda item: item[2],
        )
    ]
    return reduced_vocab, reduced_merges


def build_added_tokens(tokenizer_json: dict) -> list[dict]:
    added_tokens = []
    for entry in tokenizer_json.get("added_tokens", []):
        if entry["content"] not in SPECIALS:
            continue
        added_tokens.append(
            {
                "id": entry["id"],
                "content": entry["content"],
                "single_word": bool(entry.get("single_word", False)),
                "lstrip": bool(entry.get("lstrip", False)),
                "rstrip": bool(entry.get("rstrip", False)),
                "normalized": bool(entry.get("normalized", False)),
                "special": bool(entry.get("special", False)),
            }
        )
    return added_tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", default=Path("testdata/omnivoice/tokenizer.json"), type=Path)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer_json = json.loads((args.model / "tokenizer.json").read_text())
    reduced_vocab, reduced_merges = build_merge_closure(tokenizer_json, tokenizer)

    model = {field: tokenizer_json["model"].get(field) for field in MODEL_FIELDS}
    model["vocab"] = reduced_vocab
    model["merges"] = reduced_merges

    fixture = {
        "version": tokenizer_json.get("version", "1.0"),
        "cases": [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)} for text in CASES],
        "truncation": tokenizer_json.get("truncation"),
        "padding": tokenizer_json.get("padding"),
        "added_tokens": build_added_tokens(tokenizer_json),
        "normalizer": tokenizer_json.get("normalizer"),
        "pre_tokenizer": tokenizer_json.get("pre_tokenizer"),
        "post_processor": tokenizer_json.get("post_processor"),
        "decoder": tokenizer_json.get("decoder"),
        "model": model,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
