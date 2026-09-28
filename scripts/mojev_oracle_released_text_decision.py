#!/usr/bin/env python3
"""Pin one released MoJev CPU text decision, including packed IDs and logits.

Run only with the approved, hash-pinned checkpoint and tokenizer. This records
released behavior, including its known cross-question isolation caveat.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
MODEL_SHA = "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458"
SERVE_SHA = "7d244785d11eb5cdb490080f24f1307b2c5df8126a6b0858c4f1db341d838c8d"
FULL_SHA = "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79"
CONFIG_SHA = "1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9"
WEIGHTS_SHA = "eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50"
WEIGHTS_SIZE = 1710234304
TOKENIZER_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
TOKENIZER_CONFIG_SHA = "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: mojev_oracle_released_text_decision.py PINNED_SOURCE CHECKPOINT_DIR OUTPUT.json")
    source, checkpoint, dest = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source revision mismatch")
    for path, expected in ((source / "mojev/modeling.py", MODEL_SHA), (source / "mojev/serve.py", SERVE_SHA),
                           (source / "mojev/full.py", FULL_SHA), (checkpoint / "config.json", CONFIG_SHA),
                           (checkpoint / "tokenizer.json", TOKENIZER_SHA),
                           (checkpoint / "tokenizer_config.json", TOKENIZER_CONFIG_SHA)):
        if digest(path) != expected:
            raise SystemExit(f"asset hash mismatch: {path}")
    weights = checkpoint / "model.safetensors"
    if weights.stat().st_size != WEIGHTS_SIZE or digest(weights) != WEIGHTS_SHA:
        raise SystemExit("checkpoint size/hash mismatch")
    sys.path.insert(0, str(source))
    import torch
    from transformers import AutoTokenizer
    from mojev.modeling import PackedScorer
    from mojev.serve import option_texts, build_answer
    from mojev.full import sort_candidates, unsort, packed_collate
    from mojev.schema import Field, Schema
    from mojev.data import Example
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, use_fast=False)
    model = PackedScorer.from_pretrained(str(checkpoint), local_files_only=True, use_safetensors=True).eval()
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
    batch = packed_collate(tokenizer, schema, context_tokens=128)([Example(request["state"], tuple(0 for _ in names), tuple(menus))])
    with torch.inference_mode():
        all_logits = model(batch)[0]
    logits = [all_logits[i, :len(menu)].tolist() for i, menu in enumerate(menus)]
    answers = {}
    for i, name in enumerate(names):
        probabilities = unsort(orders[i], all_logits[i, :len(menus[i])].softmax(-1).tolist())
        answers[name] = build_answer(name, request["questions"][name], kinds[i], probabilities)
    used = int(batch["packed_mask"].sum())
    result = dict(schema=1, source_revision=REV, modeling_sha256=MODEL_SHA, serve_sha256=SERVE_SHA,
                  full_sha256=FULL_SHA, config_sha256=CONFIG_SHA, weights_sha256=WEIGHTS_SHA,
                  weights_size=WEIGHTS_SIZE, tokenizer_sha256=TOKENIZER_SHA,
                  tokenizer_config_sha256=TOKENIZER_CONFIG_SHA, request=request,
                  packed_ids=batch["packed_ids"][0].tolist(), sorted_logits=logits,
                  response=dict(model=request["model"], answers=answers,
                                usage=dict(input_tokens=used, output_tokens=0)))
    dest.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
