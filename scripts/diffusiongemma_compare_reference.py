#!/usr/bin/env python3
"""Compare DiffusionGemma reference and go-pherence run JSON outputs.

This is a utility script, not a test. It is intended for parity triage once a
Transformers reference fixture and a go-pherence run JSON are available.
"""
import argparse
import json
import pathlib
import sys


def load(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ids_from_reference(obj):
    if "output_ids" in obj and obj["output_ids"] is not None:
        return obj["output_ids"]
    if "generated" in obj:
        return obj["generated"]
    return None


def ids_from_run(obj):
    if obj.get("result") and obj["result"].get("generated") is not None:
        return obj["result"]["generated"]
    if obj.get("generated") is not None:
        return obj["generated"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=pathlib.Path, required=True)
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--prefix", type=int, default=0, help="compare only first N generated/output IDs; 0 means all")
    args = ap.parse_args()

    ref = load(args.reference)
    run = load(args.run)
    ref_ids = ids_from_reference(ref)
    run_ids = ids_from_run(run)
    result = {
        "reference": str(args.reference),
        "run": str(args.run),
        "reference_ids": len(ref_ids) if ref_ids is not None else None,
        "run_ids": len(run_ids) if run_ids is not None else None,
        "match": False,
        "first_mismatch": None,
        "error": None,
    }
    if ref_ids is None or run_ids is None:
        result["error"] = "missing output/generated IDs"
        print(json.dumps(result, indent=2))
        return 2
    n = args.prefix if args.prefix > 0 else min(len(ref_ids), len(run_ids))
    ref_cmp = ref_ids[:n]
    run_cmp = run_ids[:n]
    result["compared"] = n
    result["match"] = ref_cmp == run_cmp and (args.prefix > 0 or len(ref_ids) == len(run_ids))
    if ref_cmp != run_cmp:
        for i, (a, b) in enumerate(zip(ref_cmp, run_cmp)):
            if a != b:
                result["first_mismatch"] = {"index": i, "reference": a, "run": b}
                break
        if result["first_mismatch"] is None:
            result["first_mismatch"] = {"index": min(len(ref_cmp), len(run_cmp)), "reference": None, "run": None}
    elif args.prefix == 0 and len(ref_ids) != len(run_ids):
        result["first_mismatch"] = {"index": n, "reference_len": len(ref_ids), "run_len": len(run_ids)}
    print(json.dumps(result, indent=2))
    return 0 if result["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
