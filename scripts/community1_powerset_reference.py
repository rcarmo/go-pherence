#!/usr/bin/env python3
"""Generate small powerset conversion fixtures from checksum-pinned pyannote.

Offline developer oracle only. Requires PyTorch CPU, loads only powerset.py via
importlib (not the pyannote pipeline), and never loads weights/audio or runs a
neural model. Runs one CPU thread. No network access in this script.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

REVISION = "b749285c5cdd4636b2edc7f766f1352c8dde9369"
URL = f"https://raw.githubusercontent.com/pyannote/pyannote-audio/{REVISION}/src/pyannote/audio/utils/powerset.py"
SHA256 = "7eeb5691c20337bd24462f6a4ee2e56e279564bfef7875776df0a41f245b0325"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = Path(args.reference)
    if hashlib.sha256(reference.read_bytes()).hexdigest() != SHA256:
        raise SystemExit("pyannote powerset source checksum mismatch")

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    spec = importlib.util.spec_from_file_location("pinned_powerset", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    for speakers, active in [(1, 1), (3, 2), (4, 2), (4, 4)]:
        converter = module.Powerset(speakers, active)
        classes = converter.num_powerset_classes
        # Every class must win one row, plus complete ties and smooth overlaps.
        scores = [[0.0 if i == winner else -80.0 for i in range(classes)]
                  for winner in range(classes)]
        scores.append([-math.log(classes)] * classes)
        for frame in range(5):
            logits = [math.sin((frame + 1) * (i + 2)) * 3 for i in range(classes)]
            normalizer = math.log(sum(math.exp(x) for x in logits))
            scores.append([x - normalizer for x in logits])
        tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
        hard = converter.to_multilabel(tensor, soft=False)
        soft = converter.to_multilabel(tensor, soft=True)
        permutations = [{"slots": list(slots), "classes": list(mapping)}
                        for slots, mapping in converter.permutation_mapping.items()]
        cases.append({"speakers": speakers, "max_active": active,
                      "classes": classes, "frames": len(scores),
                      "log_scores": tensor.flatten().tolist(),
                      "hard": hard.flatten().tolist(), "soft": soft.flatten().tolist(),
                      "permutations": permutations})
    output = {"schema": 1, "reference": {"url": URL, "revision": REVISION,
              "sha256": SHA256, "pyannote_audio_version": "4.0.7", "license": "MIT",
              "torch_version": torch.__version__, "device": "cpu", "threads": 1},
              "scope": "synthetic powerset conversion/permutation; no checkpoint or diarization quality",
              "soft_absolute_tolerance": 1e-6, "cases": cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {len(cases)} synthetic powerset cases")


if __name__ == "__main__":
    main()
