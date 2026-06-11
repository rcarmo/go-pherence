#!/usr/bin/env python3
"""Compare diffusiongemmarun -json generated IDs against expected IDs."""
import argparse
import json
import pathlib
import sys


def parse_expected(text: str) -> list[int]:
    out = []
    for part in text.replace(",", " ").split():
        if part.strip():
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path, help="diffusiongemmarun -json output")
    ap.add_argument("--expected", required=True, help="comma/space-separated expected token IDs")
    ap.add_argument("--prefix", action="store_true", help="only require generated output to start with expected IDs")
    args = ap.parse_args()
    data = json.loads(args.run.read_text())
    result = data.get("result") or {}
    generated = result.get("generated") or []
    expected = parse_expected(args.expected)
    if args.prefix:
        match = generated[: len(expected)] == expected
    else:
        match = generated == expected
    report = {
        "match": match,
        "prefix": args.prefix,
        "expected": expected,
        "generated": generated,
        "error": data.get("error"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
