#!/usr/bin/env python3
"""Emit the pinned OpenJEV 0.8B text NLI CPU oracle."""
import argparse
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="local qwen3.5-0.8b-nli-v2s-long directory")
    args = ap.parse_args()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).eval()
    pairs = [
        ("A man is playing a guitar.", "Someone is making music."),
        ("The sky is blue.", "The sky is green."),
        ("A person reads a book.", "Someone is outdoors."),
        ("Which gas do plants absorb during photosynthesis?", "The correct answer is: carbon dioxide"),
    ]
    texts = [model.config.nli_template.format(premise=p.strip(), hypothesis=h.strip()) for p, h in pairs]
    enc = tok(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model.model(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        pooled = hidden[torch.arange(len(texts)), last]
        logits = model.score(pooled).float()
        probabilities = torch.softmax(logits, -1)
    out = {
        "model_pin": "4395b29714015162db6112de91c35688e6e42717",
        "model_sha256": "cf6d62a341c0c804f9a926eec71aefc9859adb28978736e757b49bce35d9b8f8",
        "template": model.config.nli_template,
        "labels": [model.config.id2label[i] for i in range(3)],
        "pairs": [],
    }
    for i, (pair, text) in enumerate(zip(pairs, texts)):
        out["pairs"].append({
            "premise": pair[0], "hypothesis": pair[1], "text": text,
            "ids": enc["input_ids"][i, :last[i] + 1].tolist(),
            "logits": logits[i].tolist(), "probabilities": probabilities[i].tolist(),
        })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
