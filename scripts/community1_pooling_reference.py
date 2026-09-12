#!/usr/bin/env python3
"""Generate pinned pyannote StatsPool synthetic CPU fixtures for WeSpeaker.

One CPU thread, no trained weights/audio/ResNet/pipeline. Source checksum must
match. Output includes interpolated support metadata separately from statistics.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

SHA = "8cb687441630e6759fb6ca545d649b41dbec1d42f869954c0a03ae191c7cbd82"
REV = "b749285c5cdd4636b2edc7f766f1352c8dde9369"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if hashlib.sha256(Path(args.reference).read_bytes()).hexdigest() != SHA:
        raise SystemExit("StatsPool source checksum mismatch")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    spec = importlib.util.spec_from_file_location("pinned_stats_pool", args.reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = []
    geometries = [(3, 7, None), (1, 2, None),
                  (3, 7, [[1]*7, [0]*7, [0, 0, 1, 0, 0, 0, 0], [1, .2, .8, 0, .4, 1, .5]]),
                  (5, 11, [[1, 0, .4, 1], [0, 1, 0, .5]]),
                  (4, 3, [[1, 0, 0, 1, 0, 1, 0, 0], [0, .3, 1, 0, .1, 0, .7, 1]]),
                  (2, 1, [[0], [1], [.3], [1e-8]]),
                  (2, 5, [[.2, .2, .2, .2, .2], [.1, .3, .7, .9, .2]]),
                  (2, 82, [[0, 1], [1, 0]]), (1, 94, [[0, 1]])]
    with torch.no_grad():
        for index, (features, frames, weights) in enumerate(geometries):
            x = (torch.sin(torch.arange(features*frames, dtype=torch.float32)*.39+index)*.5).reshape(1, features, frames)
            mask = torch.tensor(weights, dtype=torch.float32)[None] if weights is not None else None
            output = module.StatsPool()(x, mask)
            if mask is None:
                counts, sums = [frames], [float(frames)]
            else:
                aligned = torch.nn.functional.interpolate(mask, size=frames, mode="nearest")
                counts = (aligned > 0).sum(-1).flatten().tolist()
                sums = aligned.sum(-1).flatten().tolist()
            if not torch.isfinite(output).all():
                raise RuntimeError("nonfinite oracle output")
            cases.append({"config": {"Features": features, "Frames": frames,
                         "Speakers": len(weights) if weights is not None else 0,
                         "MaskFrames": len(weights[0]) if weights is not None else 0},
                         "input": x.flatten().tolist(), "masks": mask.flatten().tolist() if mask is not None else None,
                         "output": output.flatten().tolist(), "weight_sum": sums, "nonzero_frames": counts})
    index_cases = []
    for source, target in [(2, 82), (2, 94), (2, 166), (2, 4), (3, 3), (4, 11), (8, 3), (17, 125), (31, 4096), (4096, 31)]:
        indexes = torch.nn.functional.interpolate(torch.arange(source, dtype=torch.float32)[None, None], size=target, mode="nearest").to(torch.int64).flatten().tolist()
        index_cases.append({"source": source, "target": target, "indexes": indexes})
    result = {"schema": 1, "absolute_tolerance": 2e-6, "relative_tolerance": 2e-6,
              "reference": {"url": f"https://raw.githubusercontent.com/pyannote/pyannote-audio/{REV}/src/pyannote/audio/models/blocks/pooling.py",
                            "sha256": SHA, "torch": torch.__version__, "torch_commit": torch.version.git_version,
                            "cpu_threads": 1},
              "scope": "StatsPool only, not speaker embedding or mask admission quality",
              "cases": cases, "index_cases": index_cases}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(f"wrote {len(cases)} synthetic StatsPool cases")


if __name__ == "__main__":
    main()
