#!/usr/bin/env python3
"""Emit the pinned Decider 0.8B Choice/Noul/isolated-Score CPU oracle."""
import argparse
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="local Mapika/decider-0.8b model directory")
    ap.add_argument("--source", help="local Mapika/decider source checkout", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.source)
    from decider.prompt import build, letter_ids
    from decider.systemone import render_question, plan_rows, assemble, render_state
    from decider.infer import Example, Q

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True).eval()
    state = {"ticket": "I was charged twice and want a refund.", "priority": 2}
    questions = {
        "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages", "other": None}},
        "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]},
    }
    rendered = {k: render_question(v) for k, v in questions.items()}
    rows, index = plan_rows(rendered, True)

    class Keep:
        def shuffle(self, values): pass
        def sample(self, values, count): return values[:count]

    context = render_state(state)
    items = [build(Example(context, [Q(r["question"], r["options"], 0)]), tok, Keep(), max_options=255, max_ctx_tokens=32768) for r in rows]
    width = max(len(x["ids"]) for x in items)
    ids = torch.full((len(items), width), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, item in enumerate(items):
        ids[i, :len(item["ids"])] = torch.tensor(item["ids"])
        mask[i, :len(item["ids"])] = 1
    with torch.no_grad():
        output = model(input_ids=ids, attention_mask=mask).logits
    labels = letter_ids(tok)
    logits, probabilities = [], []
    for i, item in enumerate(items):
        row = output[i, item["slots"][0], labels[:item["nopts"][0]]].float()
        logits.append(row.tolist())
        probabilities.append(torch.softmax(row / 1.03, -1).tolist())
    print(json.dumps({
        "source_pin": "c4daaac28af9fea95d627015cffa2dd5a5926ee6",
        "model_pin": "1ea54127d3bd52f6d753d9257b32a6380b873907",
        "base_pin": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "state": state, "rendered_state": context, "questions": questions, "rows": rows, "index": index,
        "label_ids": labels,
        "prompts": [{"text": tok.decode(item["ids"]), "ids": item["ids"], "slot": item["slots"][0], "nopts": item["nopts"][0], "logits": logit, "probabilities": probability} for item, logit, probability in zip(items, logits, probabilities)],
        "answers": assemble(rendered, index, probabilities),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
