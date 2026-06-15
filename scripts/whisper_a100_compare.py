#!/usr/bin/env python3
"""Compare Whisper baseline output against the opt-in A100 row-scale FFN mode.

This is a validation harness for K3/A100-capable hosts. On non-riscv/non-A100
hosts the A100 env vars are harmless because the Go stubs keep the path disabled,
so the script still works as a command/prompt regression smoke. Pass --audio
multiple times to validate several clips in one run. Use --start/--duration to
slice long recordings into practical validation windows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND_ENVS = {
    "a100": {
        "WHISPER_A100_FFN_FUSED": "1",
        "WHISPER_A100_X100_PACK": "1",
        "WHISPER_A100_NATIVE_Q8": "1",
    },
    "int8": {
        "WHISPER_INT8": "1",
    },
}


def run_case(cmd: list[str], env: dict[str, str], timeout: int, output: Path | None = None) -> dict:
    if output is not None and output.exists():
        output.unlink()
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)
    file_text = ""
    if output is not None and output.exists():
        file_text = output.read_text(encoding="utf-8").strip()
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr, "output": str(output) if output is not None else "", "output_text": file_text}


def materialize_window(audio: str, idx: int, args: argparse.Namespace, out_dir: Path) -> tuple[str, str | None]:
    if args.start <= 0 and args.duration <= 0:
        return audio, None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"whisper_a100_compare_{idx:02d}_window.wav"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if args.start > 0:
        cmd += ["-ss", str(args.start)]
    if args.duration > 0:
        cmd += ["-t", str(args.duration)]
    cmd += ["-i", audio, "-ar", "16000", "-ac", "1", str(out)]
    subprocess.run(cmd, check=True)
    return str(out), str(out)


def build_cmd(args: argparse.Namespace, audio: str, baseline_output: Path | None) -> tuple[list[str], str]:
    if args.diarize_vtt:
        cmd = ["go", "run", "./cmd/audio/diarize-vtt", "-input", audio, "-output", str(baseline_output), "-model", args.model, "-size", args.size, "-task", args.task, "-language", args.language, "-max-tokens", str(args.max_tokens), "-gpu=false", "-workers", "1", "-progressive=false", "-resume=false"]
        if args.speaker_model:
            cmd.extend(["-speaker-model", args.speaker_model])
        return cmd, "output_text"
    cmd = ["go", "run", "./cmd/audio/whisper", "-audio", audio, "-model", args.model, "-size", args.size, "-task", args.task, "-language", args.language, "-max-tokens", str(args.max_tokens)]
    if args.timestamps:
        cmd.extend(["-timestamps", "-output", str(baseline_output)])
        return cmd, "output_text"
    return cmd, "stdout"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", action="append", default=[], help="Audio file to compare; repeatable (default: testdata/jfk.wav)")
    ap.add_argument("--backend", choices=sorted(BACKEND_ENVS), default="a100", help="Backend env preset to compare against baseline")
    ap.add_argument("--model", default="models/whisper-large-v3-turbo-hf/model.safetensors")
    ap.add_argument("--size", default="turbo")
    ap.add_argument("--task", default="translate", choices=("translate", "transcribe"))
    ap.add_argument("--language", default="en")
    ap.add_argument("--max-tokens", default="16")
    ap.add_argument("--timestamps", action="store_true", help="Compare standalone whisper timestamp/VTT output instead of stdout")
    ap.add_argument("--diarize-vtt", action="store_true", help="Compare cmd/audio/diarize-vtt VTT output")
    ap.add_argument("--speaker-model", default="", help="Optional speaker model passed to --diarize-vtt")
    ap.add_argument("--start", type=float, default=0.0, help="Optional start offset for every input audio")
    ap.add_argument("--duration", type=float, default=0.0, help="Optional duration window for every input audio")
    ap.add_argument("--output-dir", default="/workspace/tmp")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    audios = args.audio or ["testdata/jfk.wav"]
    out_dir = Path(args.output_dir)
    env = os.environ.copy()
    backend_env = BACKEND_ENVS[args.backend]
    compare_env = env | backend_env
    reports = []
    failures = 0

    for idx, original_audio in enumerate(audios):
        audio, window_path = materialize_window(original_audio, idx, args, out_dir)
        baseline_output = None
        a100_output = None
        if args.timestamps or args.diarize_vtt:
            out_dir.mkdir(parents=True, exist_ok=True)
            mode = "diarize" if args.diarize_vtt else "timestamps"
            baseline_output = out_dir / f"whisper_a100_compare_{idx:02d}_{mode}_baseline.vtt"
            a100_output = out_dir / f"whisper_a100_compare_{idx:02d}_{mode}_a100.vtt"
        base_cmd, compare_field = build_cmd(args, audio, baseline_output)
        baseline = run_case(base_cmd, env, args.timeout, baseline_output)
        a100_cmd = list(base_cmd)
        if compare_field == "output_text" and baseline_output is not None and a100_output is not None:
            a100_cmd[a100_cmd.index(str(baseline_output))] = str(a100_output)
        a100 = run_case(a100_cmd, compare_env, args.timeout, a100_output)
        ok = baseline["returncode"] == 0 and a100["returncode"] == 0 and baseline[compare_field] == a100[compare_field]
        if not ok:
            failures += 1
        report = {"ok": ok, "audio": original_audio, "window_audio": window_path or "", "model": args.model, "size": args.size, "task": args.task, "language": args.language, "max_tokens": args.max_tokens, "start": args.start, "duration": args.duration, "timestamps": args.timestamps, "diarize_vtt": args.diarize_vtt, "speaker_model": args.speaker_model, "backend": args.backend, "backend_env": backend_env, "compare_field": compare_field, "baseline": baseline, "backend_run": a100}
        reports.append(report)
        if not args.json:
            print("OK" if ok else "FAIL", original_audio, f"baseline={baseline[compare_field]!r}", f"{args.backend}={a100[compare_field]!r}")
            if baseline["returncode"] != 0:
                print("baseline stderr:", baseline["stderr"], file=sys.stderr)
            if a100["returncode"] != 0:
                print(f"{args.backend} stderr:", a100["stderr"], file=sys.stderr)

    if args.json:
        print(json.dumps({"ok": failures == 0, "cases": reports}, indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
