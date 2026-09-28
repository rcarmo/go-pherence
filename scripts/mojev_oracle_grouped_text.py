#!/usr/bin/env python3
"""Independent 4096-total-token reference: fresh upstream candidate forwards.

This is 64 paths of 190 tokens, not a 4096-token individual path. Go groups those
paths into bounded trees. No Go-generated scores or hidden states are used.
"""
import inspect
import json
import pathlib
import subprocess
import sys

from mojev_oracle_repaired_text_isolation import (
    REV, MODEL_SHA, CONFIG_SHA, WEIGHTS_SHA, WEIGHTS_SIZE, TRANSFORMERS_SHA, digest,
)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: mojev_oracle_grouped_text.py SOURCE CHECKPOINT OUTPUT.json")
    source, checkpoint, dest = (pathlib.Path(p).resolve() for p in sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source mismatch")
    for path, sha in ((source / "mojev/modeling.py", MODEL_SHA),
                      (checkpoint / "config.json", CONFIG_SHA),
                      (checkpoint / "model.safetensors", WEIGHTS_SHA)):
        if digest(path) != sha:
            raise SystemExit(f"asset mismatch: {path}")
    if (checkpoint / "model.safetensors").stat().st_size != WEIGHTS_SIZE:
        raise SystemExit("weight size mismatch")
    sys.path.insert(0, str(source))
    import torch
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    from mojev.modeling import PackedScorer
    if transformers.__version__ != "5.17.0" or digest(pathlib.Path(inspect.getfile(qwen))) != TRANSFORMERS_SHA:
        raise SystemExit("Transformers mismatch")
    torch.set_num_threads(2)
    model = PackedScorer.from_pretrained(str(checkpoint), local_files_only=True,
                                        use_safetensors=True).float().eval()
    ids = lambda n, seed: [100 + (i * 7 + seed) % 800 for i in range(n)]
    row = dict(state=ids(64, 4), questions=[ids(64, 5)],
               candidates=[[ids(62, 6 + i) for i in range(64)]])
    assert len(row["state"]) + len(row["questions"][0]) + sum(map(len, row["candidates"][0])) == 4096

    def run(candidate, capture):
        tokens = row["state"] + row["questions"][0] + candidate
        n, positions = len(tokens), [63, 127, len(tokens) - 1]
        state, field, option = torch.zeros((1, n)), torch.zeros((1, 1, n)), torch.zeros((1, 1, 1, n))
        state[:, :64] = 1
        field[:, :, 64:128] = 1
        option[:, :, :, 128:] = 1
        batch = dict(packed_ids=torch.tensor([tokens]), packed_mask=torch.ones((1, n), dtype=torch.bool),
                     context_span=state, field_span=field, option_span=option,
                     option_mask=torch.ones((1, 1, 1), dtype=torch.bool))
        hidden = []
        hook = model.encoder.register_forward_hook(
            lambda m, a, out: hidden.append(out.last_hidden_state[0, positions].detach().float().clone())) if capture else None
        try:
            with torch.inference_mode():
                score = float(model(batch)[0, 0, 0])
        finally:
            if hook is not None:
                hook.remove()
        return score, hidden[0].tolist() if capture else None, positions

    logits, hidden_samples = [], []
    for i, candidate in enumerate(row["candidates"][0]):
        score, hidden, positions = run(candidate, i in (0, 31, 63))
        logits.append(score)
        if hidden is not None:
            hidden_samples.append(dict(candidate=i, positions=positions, hidden=hidden))
    changed = list(row["candidates"][0][63])
    changed[0] = 1234
    changed_score, _, _ = run(changed, False)
    assert changed_score != logits[63]
    # A fresh rerun after the changed sibling checks the independent path has no
    # persistent state. The Go test separately checks every unaffected score.
    repeated, _, _ = run(row["candidates"][0][0], False)
    assert repeated == logits[0]
    dest.write_text(json.dumps(dict(schema=1, policy="f32-branch-local-positions-full-tree-causal-linear",
                                   source_revision=REV, modeling_sha256=MODEL_SHA, config_sha256=CONFIG_SHA,
                                   weights_sha256=WEIGHTS_SHA, weights_size=WEIGHTS_SIZE,
                                   transformers_version=transformers.__version__, torch_version=torch.__version__,
                                   transformers_qwen_sha256=TRANSFORMERS_SHA, row=row, logits=[logits],
                                   hidden_samples=hidden_samples,
                                   sibling_changed=dict(candidate=63, token=1234, logit=changed_score)),
                               separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
