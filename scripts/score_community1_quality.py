#!/usr/bin/env python3
"""Score a hash-pinned trained Community-1 corpus result against RTTM.

This reports absolute full/exclusive DER and JER. It does not compare the Go
result to a reference-system run and therefore never marks a result qualified.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("community1_corpus", ROOT / "score_community1_corpus.py")
CORPUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORPUS)

WRAPPER_KEYS = ("Schema", "InputSHA256", "InputSamples", "ElapsedNanos", "Config", "Result")


def load_result(path, expected_audio_hash, expected_samples):
    data = CORPUS.read_json(path)
    CORPUS.exact_keys(data, WRAPPER_KEYS)
    if CORPUS.exact_int(data["Schema"], "result schema") != 1:
        raise ValueError("unsupported trained corpus result schema")
    if data["InputSHA256"] != expected_audio_hash or not CORPUS.HEX64.fullmatch(data["InputSHA256"]):
        raise ValueError("trained corpus input hash changed")
    if CORPUS.exact_int(data["InputSamples"], "input samples", 1, 14400 * 16000) != expected_samples:
        raise ValueError("trained corpus input samples changed")
    CORPUS.exact_int(data["ElapsedNanos"], "elapsed nanoseconds", 1)
    CORPUS.exact_keys(data["Config"], CORPUS.GO_CONFIG_KEYS)
    CORPUS.exact_keys(data["Result"], CORPUS.GO_RESULT_KEYS)
    post = data["Result"]["Postprocess"]
    CORPUS.exact_keys(post, CORPUS.GO_POSTPROCESS_KEYS)
    timeline = post["Timeline"]
    CORPUS.exact_keys(timeline, CORPUS.GO_TIMELINE_KEYS)
    tie_policy = CORPUS.exact_int(data["Config"]["TiePolicy"], "tie policy")
    if tie_policy not in (0, 1):
        raise ValueError("invalid tie policy")
    ambiguous = timeline["AmbiguousFrames"]
    if not isinstance(ambiguous, list) or any(CORPUS.exact_int(value, "ambiguous frame", 0) < 0 for value in ambiguous):
        raise ValueError("invalid ambiguous frames")
    if tie_policy == 0 and ambiguous:
        raise ValueError("strict result retained ambiguous frames")
    extent = max(expected_samples / 16000, max((float(window["Start"] + window["Samples"]) / 16000 for window in data["Result"]["Windows"]), default=0)) + 10
    turns = {
        "full": CORPUS.parse_turns(post["FullTurns"], extent, True),
        "exclusive": CORPUS.parse_turns(post["ExclusiveTurns"], extent, True, True, bool(data["Config"]["MinDurationOff"])),
    }
    return data, turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--go-result", required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--rttm", required=True)
    parser.add_argument("--rttm-sha256", required=True)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--uem-start", type=float, default=0)
    parser.add_argument("--uem-end", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("output already exists")
    if not CORPUS.HEX64.fullmatch(args.audio_sha256) or not CORPUS.HEX64.fullmatch(args.rttm_sha256):
        raise ValueError("invalid expected SHA-256")
    if CORPUS.sha256(args.rttm) != args.rttm_sha256:
        raise ValueError("RTTM checksum changed")
    if args.uem_start < 0 or args.uem_start >= args.uem_end or args.uem_end > args.samples / 16000:
        raise ValueError("invalid UEM")
    CORPUS.verify_rttm(args.rttm, {"sha256": args.rttm_sha256, "uri": args.uri}, args.samples / 16000)
    data, turns = load_result(args.go_result, args.audio_sha256, args.samples)
    collars = [0.0, 0.25]
    scores = {kind: CORPUS.score_turns(args.rttm, args.uri, [[args.uem_start, args.uem_end]], values, collars) for kind, values in turns.items()}
    result = {
        "schema": 1,
        "go_result_sha256": CORPUS.sha256(args.go_result),
        "audio_sha256": args.audio_sha256,
        "rttm_sha256": args.rttm_sha256,
        "samples": args.samples,
        "uem": [args.uem_start, args.uem_end],
        "collars": collars,
        "tie_policy": data["Config"]["TiePolicy"],
        "elapsed_ns": data["ElapsedNanos"],
        "windows": len(data["Result"]["Windows"]),
        "training_rows": data["Result"]["Postprocess"]["TrainingRows"],
        "clusters": data["Result"]["Postprocess"]["Clusters"],
        "ambiguous_frames": len(data["Result"]["Postprocess"]["Timeline"]["AmbiguousFrames"]),
        "scores": scores,
        "qualified": False,
        "scope": "absolute pilot DER/JER only; no ratified corpus budget or pinned reference-system delta",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
