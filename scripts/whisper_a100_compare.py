#!/usr/bin/env python3
"""Compare Whisper baseline output against the opt-in A100 row-scale FFN mode.

This is a validation harness for K3/A100-capable hosts. On non-riscv/non-A100
hosts the A100 env vars are harmless because the Go stubs keep the path disabled,
so the script still works as a command/prompt regression smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

A100_ENV = {
    "WHISPER_A100_FFN_FUSED": "1",
    "WHISPER_A100_X100_PACK": "1",
    "WHISPER_A100_NATIVE_Q8": "1",
}


def run_case(cmd: list[str], env: dict[str, str], timeout: int) -> dict:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default="testdata/jfk.wav")
    ap.add_argument("--model", default="models/whisper-large-v3-turbo-hf/model.safetensors")
    ap.add_argument("--size", default="turbo")
    ap.add_argument("--task", default="translate", choices=("translate", "transcribe"))
    ap.add_argument("--language", default="en")
    ap.add_argument("--max-tokens", default="16")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    base_cmd = [
        "go",
        "run",
        "./cmd/audio/whisper",
        "-audio",
        args.audio,
        "-model",
        args.model,
        "-size",
        args.size,
        "-task",
        args.task,
        "-language",
        args.language,
        "-max-tokens",
        str(args.max_tokens),
    ]
    env = os.environ.copy()
    baseline = run_case(base_cmd, env, args.timeout)
    a100_env = env | A100_ENV
    a100 = run_case(base_cmd, a100_env, args.timeout)

    ok = baseline["returncode"] == 0 and a100["returncode"] == 0 and baseline["stdout"] == a100["stdout"]
    report = {
        "ok": ok,
        "audio": args.audio,
        "model": args.model,
        "size": args.size,
        "task": args.task,
        "language": args.language,
        "max_tokens": args.max_tokens,
        "a100_env": A100_ENV,
        "baseline": baseline,
        "a100": a100,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("OK" if ok else "FAIL", f"baseline={baseline['stdout']!r}", f"a100={a100['stdout']!r}")
        if baseline["returncode"] != 0:
            print("baseline stderr:", baseline["stderr"], file=sys.stderr)
        if a100["returncode"] != 0:
            print("a100 stderr:", a100["stderr"], file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
