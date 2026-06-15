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


def ids_from_reference(obj, *, generated_only: bool = True):
    """Return reference token IDs.

    HF/llama.cpp-style fixtures usually store output_ids as prompt+response, while
    go-pherence run JSON stores result.generated as response-only.  Default to the
    response slice when input_ids are available so the comparison is a real
    generated-output parity check instead of a prompt-prefix length mismatch.
    """
    if "output_ids" in obj and obj["output_ids"] is not None:
        output = obj["output_ids"]
        prompt = obj.get("input_ids")
        if generated_only and prompt is not None and output[: len(prompt)] == prompt:
            return output[len(prompt) :]
        return output
    if "generated" in obj:
        return obj["generated"]
    if "response_ids" in obj:
        return obj["response_ids"]
    return None


def ids_from_run(obj, *, canvas: bool = False):
    if obj.get("result"):
        result = obj["result"]
        if canvas and result.get("canvases"):
            return result["canvases"][0].get("canvas")
        if result.get("generated") is not None:
            return result["generated"]
    if obj.get("generated") is not None:
        return obj["generated"]
    return None


def prompt_ids_from_run(obj):
    if obj.get("prompt_ids") is not None:
        return obj["prompt_ids"]
    return obj.get("input_ids")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=pathlib.Path, required=True)
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--prefix", type=int, default=0, help="compare only first N generated/output IDs; 0 means all")
    ap.add_argument("--full-output", action="store_true", help="compare reference output_ids including prompt prefix instead of response-only")
    ap.add_argument("--run-canvas", action="store_true", help="compare against the first raw Go canvas instead of trimmed result.generated")
    args = ap.parse_args()

    ref = load(args.reference)
    run = load(args.run)
    ref_ids = ids_from_reference(ref, generated_only=not args.full_output)
    run_ids = ids_from_run(run, canvas=args.run_canvas)
    ref_prompt = ref.get("input_ids")
    run_prompt = prompt_ids_from_run(run)
    result = {
        "reference": str(args.reference),
        "run": str(args.run),
        "mode": "full_output" if args.full_output else "generated_response",
        "run_source": "first_canvas" if args.run_canvas else "generated",
        "reference_ids": len(ref_ids) if ref_ids is not None else None,
        "run_ids": len(run_ids) if run_ids is not None else None,
        "prompt_match": (ref_prompt == run_prompt) if ref_prompt is not None and run_prompt is not None else None,
        "match": False,
        "first_mismatch": None,
        "error": None,
    }
    if ref_ids is None or run_ids is None:
        result["error"] = "missing output/generated IDs"
        print(json.dumps(result, indent=2))
        return 2
    if run.get("result") and run["result"].get("canvases"):
        first_canvas = run["result"]["canvases"][0]
        result["run_first_canvas"] = {
            "len": len(first_canvas.get("canvas", [])),
            "trim_cut": first_canvas.get("trim_cut"),
            "trim_reason": first_canvas.get("trim_reason", ""),
            "steps": len(first_canvas.get("steps", [])),
        }
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
