#!/usr/bin/env python3
"""Independent 512-token repaired F32 reference, using only existing assets.

No Go output participates. Each candidate runs with fresh state and local
positions through pinned upstream Torch/Transformers. Hidden output is sampled
at the last state, question and candidate rows to bound fixture size.
"""
import copy
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
        raise SystemExit("usage: mojev_oracle_long_text.py SOURCE CHECKPOINT OUTPUT.json")
    source, checkpoint, dest = (pathlib.Path(p).resolve() for p in sys.argv[1:])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("source mismatch")
    for path, sha in ((source / "mojev/modeling.py", MODEL_SHA),
                      (checkpoint / "config.json", CONFIG_SHA),
                      (checkpoint / "model.safetensors", WEIGHTS_SHA)):
        if digest(path) != sha:
            raise SystemExit(f"asset mismatch: {path}")
    if (checkpoint / "model.safetensors").stat().st_size != WEIGHTS_SIZE:
        raise SystemExit("weights size mismatch")
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
    base = dict(state=ids(128, 0), questions=[ids(128, 1)],
                candidates=[[ids(256, 2), ids(256, 3)]])
    shared = copy.deepcopy(base)
    shared["candidates"][0] = [c[:128] for c in shared["candidates"][0]]
    cases = {"max_path": base, "shared_tree": shared}
    changed = copy.deepcopy(base)
    changed["candidates"][0][1][0] = 1234
    cases["sibling_token"] = changed
    shortened = copy.deepcopy(base)
    shortened["candidates"][0][1] = shortened["candidates"][0][1][:127]
    cases["sibling_length"] = shortened
    reversed_row = copy.deepcopy(base)
    reversed_row["candidates"][0].reverse()
    cases["candidate_order"] = reversed_row

    results = {}
    for name, row in cases.items():
        scores, hidden, sampled_positions = [], [], []
        for candidate in row["candidates"][0]:
            ns, nq = len(row["state"]), len(row["questions"][0])
            tokens = row["state"] + row["questions"][0] + candidate
            n = len(tokens)
            state, field, option = torch.zeros((1, n)), torch.zeros((1, 1, n)), torch.zeros((1, 1, 1, n))
            state[:, :ns] = 1
            field[:, :, ns:ns+nq] = 1
            option[:, :, :, ns+nq:] = 1
            batch = dict(packed_ids=torch.tensor([tokens]), packed_mask=torch.ones((1, n), dtype=torch.bool),
                         context_span=state, field_span=field, option_span=option,
                         option_mask=torch.ones((1, 1, 1), dtype=torch.bool))
            captured = []
            positions = [ns - 1, ns + nq - 1, n - 1]
            hook = model.encoder.register_forward_hook(
                lambda m, a, out: captured.append(out.last_hidden_state[0, positions].detach().float().clone()))
            try:
                with torch.inference_mode():
                    scores.append(float(model(batch)[0, 0, 0]))
            finally:
                hook.remove()
            hidden.append(captured[0].tolist())
            sampled_positions.append(positions)
        results[name] = dict(row=row, logits=[scores], hidden=hidden, hidden_positions=sampled_positions)
    for name in ("sibling_token", "sibling_length"):
        assert results[name]["logits"][0][0] == results["max_path"]["logits"][0][0]
        assert results[name]["hidden"][0] == results["max_path"]["hidden"][0]
    assert results["candidate_order"]["logits"][0] == list(reversed(results["max_path"]["logits"][0]))
    dest.write_text(json.dumps(dict(schema=1, policy="f32-branch-local-positions-full-tree-causal-linear",
                                   source_revision=REV, modeling_sha256=MODEL_SHA, config_sha256=CONFIG_SHA,
                                   weights_sha256=WEIGHTS_SHA, weights_size=WEIGHTS_SIZE,
                                   transformers_version=transformers.__version__, torch_version=torch.__version__,
                                   transformers_qwen_sha256=TRANSFORMERS_SHA, cases=results),
                               separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
