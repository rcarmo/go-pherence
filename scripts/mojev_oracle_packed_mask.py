#!/usr/bin/env python3
"""Record synthetic MoJev additive-tree and packed-padding mask observations.

Run against the pinned MIT source with CPU torch and transformers. No model weights.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

REV = "a74d58cd19ec573e83e8e27f9fecd837b8d830fb"
MODEL_SHA = "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458"


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: mojev_oracle_packed_mask.py PINNED_SOURCE OUTPUT.json")
    source, dest = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != REV:
        raise SystemExit("wrong source revision")
    if hashlib.sha256((source / "mojev/modeling.py").read_bytes()).hexdigest() != MODEL_SHA:
        raise SystemExit("source model hash mismatch")
    sys.path.insert(0, str(source))
    import torch
    from mojev.modeling import PackedScorer

    cases = []
    for name, state, packed in (
        ("padded_row", [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]),
        # Stress the column filter even if a caller's span claims a padding token.
        # The upstream collator never produces this inconsistent input.
        ("span_claims_padding_column", [1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 1, 1, 1, 1, 1, 0]),
    ):
        question = [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]]
        candidate = [[[0, 0, 0, 0, 1, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 1, 0]]]
        mask = PackedScorer.build_mask(None, torch.tensor([state], dtype=torch.float32),
                                       torch.tensor([question], dtype=torch.float32),
                                       torch.tensor([candidate], dtype=torch.float32))
        length = len(state)
        floor = torch.finfo(torch.float32).min
        keep = torch.tensor(packed, dtype=torch.bool)[None, None, None, :] | torch.eye(length, dtype=torch.bool)[None, None]
        mask = mask.masked_fill(~keep, floor)
        if mask.shape != (1, 1, length, length) or not bool(((mask == 0) | (mask == floor)).all()):
            raise SystemExit("unexpected additive-mask values")
        if not bool((mask[0, 0] == 0).any(-1).all()):
            raise SystemExit("fully masked row")
        cases.append(dict(name=name, state=state, questions=question, candidates=candidate,
                          packed=packed, allowed=(mask[0, 0] == 0).int().tolist()))
    result = dict(schema=1, source_repo="MoLeMo-Lab/mojev", source_revision=REV,
                  modeling_sha256=MODEL_SHA, masked_float32_bits="ff7fffff", cases=cases)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
