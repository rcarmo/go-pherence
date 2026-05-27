#!/usr/bin/env python3
"""Run a JSON speakercheck validation suite.

Suite format:

{
  "cases": [
    {
      "name": "jfk-single",
      "input": "testdata/jfk.wav",
      "speaker_model": "models/speaker-ecapa-voxceleb.safetensors",
      "start": 0,
      "duration": 0,
      "threshold": 0.3,
      "context": 0.5,
      "expect": [1, 1, 1, 1, 1]
    }
  ]
}

The runner shells out to `go run ./cmd/speakercheck -json` and fails if any
case exits non-zero or reports pairwise_score below --min-pairwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("suite", help="JSON suite file")
    ap.add_argument("--min-pairwise", type=float, default=1.0, help="Minimum acceptable pairwise score for labeled cases")
    ap.add_argument("--go", default="go", help="Go executable")
    args = ap.parse_args()

    suite_path = Path(args.suite)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = suite.get("cases", [])
    if not cases:
        print(f"{suite_path}: no cases", file=sys.stderr)
        return 2

    failures = 0
    for case in cases:
        name = case.get("name") or case.get("input") or "case"
        cmd = [
            args.go,
            "run",
            "./cmd/speakercheck",
            "-input",
            str(case["input"]),
            "-speaker-model",
            str(case.get("speaker_model", "models/speaker-ecapa-voxceleb.safetensors")),
            "-threshold",
            str(case.get("threshold", 0.3)),
            "-context",
            str(case.get("context", 0.5)),
            "-start",
            str(case.get("start", 0)),
            "-duration",
            str(case.get("duration", 0)),
            "-sims=false",
            "-json",
        ]
        if "expect" in case:
            cmd.extend(["-expect", ",".join(str(int(x)) for x in case["expect"])])
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            failures += 1
            print(f"FAIL {name}: command exited {proc.returncode}", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr.strip(), file=sys.stderr)
            if proc.stdout:
                print(proc.stdout.strip(), file=sys.stderr)
            continue
        report = json.loads(proc.stdout)
        score = report.get("score") or {}
        pairwise = score.get("pairwise_score")
        speakers = len(report.get("counts", {}))
        segments = len(report.get("segments", []))
        if pairwise is not None and pairwise < args.min_pairwise:
            failures += 1
            print(f"FAIL {name}: pairwise_score={pairwise:.3f} < {args.min_pairwise:.3f}", file=sys.stderr)
        else:
            suffix = f" pairwise={pairwise:.3f}" if pairwise is not None else ""
            print(f"OK {name}: segments={segments} speakers={speakers}{suffix}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
