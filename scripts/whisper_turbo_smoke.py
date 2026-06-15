#!/usr/bin/env python3
"""Run quick Whisper large-v3-turbo command smokes.

This script exercises the default Go audio commands without requiring the Python
Transformers stack. It is intentionally lightweight and intended for validation
of prompt/default wiring, not quality scoring.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], timeout: int) -> dict:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default="testdata/jfk.wav", help="Audio fixture to smoke")
    ap.add_argument("--output-dir", default="/workspace/tmp", help="Directory for temporary VTT outputs")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--json", action="store_true", help="Emit full JSON report")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = args.audio
    report = {"audio": audio, "cases": []}

    cases = [
        {
            "name": "standalone_translate_short",
            "cmd": ["go", "run", "./cmd/audio/whisper", "-audio", audio, "-task", "translate", "-language", "en", "-max-tokens", "4"],
            "expect_stdout_contains": "And",
        },
        {
            "name": "standalone_timestamp_vtt",
            "cmd": ["go", "run", "./cmd/audio/whisper", "-audio", audio, "-timestamps", "-task", "translate", "-language", "en", "-max-tokens", "8", "-output", str(out_dir / "whisper_turbo_smoke.vtt")],
            "expect_file": str(out_dir / "whisper_turbo_smoke.vtt"),
            "expect_file_contains": "WEBVTT",
        },
        {
            "name": "diarize_vtt_default_turbo",
            "cmd": ["go", "run", "./cmd/audio/diarize-vtt", "-input", audio, "-output", str(out_dir / "diarize_turbo_smoke.vtt"), "-gpu=false", "-workers", "1", "-max-tokens", "8", "-progressive=false", "-resume=false"],
            "expect_file": str(out_dir / "diarize_turbo_smoke.vtt"),
            "expect_file_contains": "WEBVTT",
        },
    ]

    failures = 0
    for case in cases:
        result = run(case["cmd"], args.timeout)
        ok = result["returncode"] == 0
        reason = ""
        if ok and case.get("expect_stdout_contains") and case["expect_stdout_contains"] not in result["stdout"]:
            ok = False
            reason = f"stdout missing {case['expect_stdout_contains']!r}"
        if ok and case.get("expect_file"):
            p = Path(case["expect_file"])
            if not p.exists():
                ok = False
                reason = f"missing output file {p}"
            elif case.get("expect_file_contains") and case["expect_file_contains"] not in p.read_text(encoding="utf-8"):
                ok = False
                reason = f"file {p} missing {case['expect_file_contains']!r}"
        if not ok:
            failures += 1
        report["cases"].append({"name": case["name"], "ok": ok, "reason": reason, **result})
        print(("OK" if ok else "FAIL"), case["name"], reason, file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
