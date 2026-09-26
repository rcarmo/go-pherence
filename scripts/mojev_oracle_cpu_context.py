#!/usr/bin/env python3
"""Independent CPU F32 513/1024/4096 branch reference using pinned assets.

No Go output or benchmark labels participate. The encoder-only case reaches
position4095 but is not a valid public two-option request. The max-request case
uses exactly4096 total tokens; shared-tree covers unequal sibling lengths.
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
        raise SystemExit("usage: mojev_oracle_cpu_context.py SOURCE CHECKPOINT OUTPUT.json")
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
    if torch.__version__ != "2.14.0+cpu":
        raise SystemExit("CPU-only Torch2.14.0 required")
    torch.set_num_threads(2)
    model = PackedScorer.from_pretrained(str(checkpoint), local_files_only=True,
                                        use_safetensors=True).float().eval()
    ids = lambda n, seed: [100 + (i * 7 + seed) % 800 for i in range(n)]
    cases = {
        "path_513": dict(state=ids(256, 0), questions=[ids(128, 1)],
                         candidates=[[ids(129, 2), ids(129, 3)]]),
        "path_1024": dict(state=ids(512, 0), questions=[ids(256, 1)],
                          candidates=[[ids(256, 2), ids(256, 3)]]),
        "max_request": dict(state=ids(3070, 0), questions=[ids(1024, 1)],
                            candidates=[[ids(1, 2), ids(1, 3)]]),
        "shared_tree": dict(state=ids(512, 0), questions=[ids(256, 1)],
                            candidates=[[ids(1024, 2), ids(2304, 3)]]),
        "encoder_4096": dict(state=ids(3071, 0), questions=[ids(1024, 1)],
                             candidates=[[ids(1, 2)]]),
    }
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
            print(f"{name}: path={n}, candidate={len(candidate)}", file=sys.stderr, flush=True)
        results[name] = dict(row=row, logits=[scores], hidden=hidden, hidden_positions=sampled_positions)
    dest.write_text(json.dumps(dict(schema=1, policy="f32-branch-local-positions-full-tree-causal-linear",
                                   source_revision=REV, modeling_sha256=MODEL_SHA, config_sha256=CONFIG_SHA,
                                   weights_sha256=WEIGHTS_SHA, weights_size=WEIGHTS_SIZE,
                                   transformers_version=transformers.__version__, torch_version=torch.__version__,
                                   transformers_qwen_sha256=TRANSFORMERS_SHA, cases=results),
                               separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
