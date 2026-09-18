#!/usr/bin/env python3
"""Validate and score one hash-pinned Community-1 corpus case.

This tool reads saved turns only. It performs no neural inference and downloads
nothing. Asset hashes and geometry are checked before pyannote.metrics imports.
"""
import argparse
import gzip
import hashlib
import json
import math
import re
import wave
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-f]{64}$")
CASE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
MAX_JSON_BYTES = 256 << 20
MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
GO_CONFIG_KEYS = ("WindowSamples", "StepSamples", "MinimumEmbeddingSamples", "ExcludeOverlap", "MinSpeakers", "MaxSpeakers", "NumSpeakers", "AHCThreshold", "Fa", "Fb", "MinDurationOff", "Constrained", "TiePolicy")
GO_RESULT_KEYS = ("Windows", "Grid", "LocalSpeakers", "EmbeddingDimension", "Segmentations", "Embeddings", "SelectedFrames", "WeightSum", "NonzeroFrames", "UsedOverlapExcluded", "Postprocess")
GO_POSTPROCESS_KEYS = ("Path", "TrainingRows", "TrainingChunks", "TrainingSpeakers", "Clusters", "InitialLabels", "HardLabels", "Centroids", "SoftScores", "ConstraintSatisfied", "Timeline", "FullTurns", "ExclusiveTurns")
GO_TIMELINE_KEYS = ("Frames", "Classes", "Start", "FrameDuration", "FrameStep", "Counts", "Activations", "Full", "Exclusive", "AmbiguousFrames")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON input too large: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"expanded JSON input too large: {path}")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value: {value}")))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    keys, allowed = set(value), set(required) | set(optional)
    if keys - allowed or set(required) - keys:
        raise ValueError(f"invalid keys: got {sorted(keys)}, require {sorted(required)}, allow {sorted(optional)}")


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid {name}")
    return float(value)


def exact_int(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid {name}")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"invalid {name}")
    return value


def validate_manifest(value):
    exact_keys(value, ("schema", "model_revision", "cases"))
    if exact_int(value["schema"], "schema") != 1 or value["model_revision"] != MODEL_REVISION:
        raise ValueError("unsupported corpus manifest identity")
    cases = value["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 256:
        raise ValueError("invalid corpus case count")
    seen = set()
    for case in cases:
        exact_keys(case, ("id", "role", "corpus", "split", "language", "licence", "audio", "rttm", "uem", "turn_extent_end", "go_result", "reference", "collars", "tie_policy", "exclusive_may_overlap", "gates"))
        case_id = case["id"]
        if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id) or case_id in seen:
            raise ValueError("invalid or duplicate corpus case id")
        seen.add(case_id)
        if case["role"] not in ("diagnostic", "qualifying"):
            raise ValueError(f"invalid role for {case_id}")
        for key in ("corpus", "split", "language"):
            if not isinstance(case[key], str) or not 1 <= len(case[key]) <= 128:
                raise ValueError(f"invalid {key} for {case_id}")
        licence = case["licence"]
        exact_keys(licence, ("spdx", "url", "redistribution"))
        if not all(isinstance(licence[k], str) and licence[k] for k in licence):
            raise ValueError(f"invalid licence for {case_id}")
        if licence["redistribution"] not in ("verified", "unverified"):
            raise ValueError(f"invalid redistribution status for {case_id}")
        if case["role"] == "qualifying" and licence["redistribution"] != "verified":
            raise ValueError(f"qualifying case has unverified redistribution: {case_id}")
        audio = case["audio"]
        exact_keys(audio, ("sha256", "sample_rate", "channels", "samples"))
        if not HEX64.fullmatch(audio["sha256"]):
            raise ValueError(f"invalid audio hash for {case_id}")
        if exact_int(audio["sample_rate"], "sample rate") != 16000 or exact_int(audio["channels"], "channels") != 1 or not 1 <= exact_int(audio["samples"], "samples") <= 16000 * 14400:
            raise ValueError(f"unsupported canonical audio geometry for {case_id}")
        rttm = case["rttm"]
        exact_keys(rttm, ("sha256", "uri"))
        if not HEX64.fullmatch(rttm["sha256"]) or not isinstance(rttm["uri"], str) or not rttm["uri"] or any(c.isspace() for c in rttm["uri"]):
            raise ValueError(f"invalid RTTM identity for {case_id}")
        for result_key, result_format in (("go_result", "community1-go-result-v1"), ("reference", "community1-reference-v1")):
            result = case[result_key]
            exact_keys(result, ("sha256", "format"))
            if not isinstance(result["sha256"], str) or not HEX64.fullmatch(result["sha256"]) or result["format"] != result_format:
                raise ValueError(f"invalid {result_key} identity for {case_id}")
        duration = audio["samples"] / 16000
        uem = case["uem"]
        if not isinstance(uem, list) or not 1 <= len(uem) <= 64:
            raise ValueError(f"invalid UEM for {case_id}")
        previous = 0.0
        for index, interval in enumerate(uem):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"invalid UEM interval for {case_id}")
            start, end = finite_number(interval[0], "UEM start"), finite_number(interval[1], "UEM end")
            if start < 0 or start >= end or end > duration or (index and start < previous):
                raise ValueError(f"invalid UEM extent for {case_id}")
            previous = end
        turn_extent_end = finite_number(case["turn_extent_end"], "turn extent end")
        if turn_extent_end < previous or turn_extent_end > duration + 30:
            raise ValueError(f"invalid turn extent for {case_id}")
        collars = case["collars"]
        if not isinstance(collars, list) or not 1 <= len(collars) <= 8:
            raise ValueError(f"invalid collars for {case_id}")
        parsed = [finite_number(v, "collar") for v in collars]
        if parsed != sorted(set(parsed)) or parsed[0] < 0 or parsed[-1] > 2:
            raise ValueError(f"invalid collars for {case_id}")
        if exact_int(case["tie_policy"], "tie policy") not in (0, 1):
            raise ValueError(f"invalid tie policy for {case_id}")
        if not isinstance(case["exclusive_may_overlap"], bool):
            raise ValueError(f"invalid exclusive overlap policy for {case_id}")
        gates = case["gates"]
        exact_keys(gates, ("max_abs_der_delta_pp", "max_abs_jer_delta_pp"))
        for key, threshold in gates.items():
            if finite_number(threshold, key) < 0:
                raise ValueError(f"invalid gate for {case_id}")
    return value


def select_case(manifest, case_id):
    matches = [case for case in manifest["cases"] if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"unknown corpus case: {case_id}")
    return matches[0]


def verify_wav(path, spec):
    if sha256(path) != spec["sha256"]:
        raise ValueError("audio checksum changed")
    with wave.open(str(path), "rb") as wav:
        geometry = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype())
    expected = (spec["channels"], 2, spec["sample_rate"], spec["samples"], "NONE")
    if geometry != expected:
        raise ValueError(f"canonical WAV geometry changed: {geometry}")


def verify_rttm(path, spec, duration):
    if sha256(path) != spec["sha256"]:
        raise ValueError("RTTM checksum changed")
    rows = 0
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER" or fields[1] != spec["uri"]:
            raise ValueError(f"invalid RTTM row {number}")
        start, length = finite_number(float(fields[3]), "RTTM start"), finite_number(float(fields[4]), "RTTM duration")
        if start < 0 or length <= 0 or start + length > duration + 1e-9:
            raise ValueError(f"RTTM extent outside audio at row {number}")
        rows += 1
    if rows == 0:
        raise ValueError("empty RTTM")


def parse_turns(values, turn_extent_end, go_format, exclusive=False, may_overlap=False):
    if not isinstance(values, list) or len(values) > 100000:
        raise ValueError("invalid turn list")
    turns = []
    previous_key = None
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("invalid turn")
        keys = ("Start", "End", "Speaker") if go_format else ("start", "end", "speaker")
        exact_keys(value, keys)
        start, end = finite_number(value[keys[0]], "turn start"), finite_number(value[keys[1]], "turn end")
        speaker = value[keys[2]]
        if start < 0 or start >= end or end > turn_extent_end + 1e-6:
            raise ValueError("turn outside declared extent")
        if go_format:
            if isinstance(speaker, bool) or not isinstance(speaker, int) or not 0 <= speaker <= 63:
                raise ValueError("invalid Go speaker")
            order_speaker, speaker = speaker, str(speaker)
        elif not isinstance(speaker, str) or not speaker or len(speaker) > 128:
            raise ValueError("invalid reference speaker")
        else:
            order_speaker = speaker
        key = (start, end, order_speaker)
        if previous_key is not None and key < previous_key:
            raise ValueError("turns not sorted")
        previous_key = key
        turns.append((start, end, speaker))
    if exclusive and not may_overlap:
        previous_end = 0.0
        for start, end, _ in turns:
            if start < previous_end - 1e-12:
                raise ValueError("exclusive turns overlap")
            previous_end = max(previous_end, end)
    return turns


def load_saved_turns(path, duration, go_format, expected_hash=None, tie_policy=None, exclusive_may_overlap=False):
    if expected_hash and sha256(path) != expected_hash:
        raise ValueError("saved result checksum changed")
    data = read_json(path)
    if go_format:
        try:
            exact_keys(data, ("Config", "Result"))
            exact_keys(data["Config"], GO_CONFIG_KEYS)
            exact_keys(data["Result"], GO_RESULT_KEYS)
            if exact_int(data["Config"]["TiePolicy"], "Go tie policy") != tie_policy:
                raise ValueError("Go result tie policy changed")
            post = data["Result"]["Postprocess"]
            exact_keys(post, GO_POSTPROCESS_KEYS)
            exact_keys(post["Timeline"], GO_TIMELINE_KEYS)
            exact_int(post["TrainingRows"], "training rows", 0, 4096)
            exact_int(post["Clusters"], "clusters", 0, 64)
            ambiguous = post["Timeline"]["AmbiguousFrames"]
            if not isinstance(ambiguous, list) or len(ambiguous) > 100000 or any(exact_int(v, "ambiguous frame", 0) < 0 for v in ambiguous):
                raise ValueError("invalid ambiguous frames")
            return {"full": parse_turns(post["FullTurns"], duration, True), "exclusive": parse_turns(post["ExclusiveTurns"], duration, True, True, exclusive_may_overlap)}, data
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid Go result shape") from exc
    try:
        exact_keys(data, ("schema", "model_revision", "source_hashes", "threads", "mkldnn", "batch_sizes", "constrained", "minimum_embedding_samples", "full", "exclusive", "artifacts"), ("num_speakers",))
        if "num_speakers" in data:
            exact_int(data["num_speakers"], "reference speaker count", 1, 64)
        if exact_int(data["schema"], "reference schema") != 1 or data["model_revision"] != MODEL_REVISION:
            raise ValueError("saved reference identity changed")
        return {"full": parse_turns(data["full"], duration, False), "exclusive": parse_turns(data["exclusive"], duration, False, True, exclusive_may_overlap)}, data
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid reference result shape") from exc


def score_turns(reference_rttm, uri, uem_intervals, turns, collars):
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    reference = load_rttm(reference_rttm)[uri]
    uem = Timeline([Segment(start, end) for start, end in uem_intervals], uri=uri)
    hypothesis = Annotation(uri=uri)
    for index, (start, end, speaker) in enumerate(turns):
        hypothesis[Segment(start, end), index] = speaker
    scores = []
    for collar in collars:
        der = DiarizationErrorRate(collar=collar, skip_overlap=False)(reference, hypothesis, uem=uem, detailed=True)
        jer = JaccardErrorRate(collar=collar, skip_overlap=False)(reference, hypothesis, uem=uem, detailed=True)
        scores.append({"collar": collar, "der": {key: float(value) for key, value in der.items()}, "jer": {key: float(value) for key, value in jer.items()}})
    return scores


def metric_value(item, metric):
    key = "diarization error rate" if metric == "der" else "jaccard error rate"
    return item[metric][key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case")
    parser.add_argument("--audio")
    parser.add_argument("--rttm")
    parser.add_argument("--go-result")
    parser.add_argument("--reference")
    parser.add_argument("--output")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = validate_manifest(read_json(args.manifest))
    if args.validate_only:
        print(json.dumps({"schema": 1, "manifest_sha256": sha256(args.manifest), "cases": len(manifest["cases"]), "valid": True}, sort_keys=True))
        return
    if not all((args.case, args.audio, args.rttm, args.go_result, args.reference, args.output)):
        parser.error("scoring requires --case, --audio, --rttm, --go-result, --reference and --output")
    output = Path(args.output)
    if output.exists():
        raise ValueError("output already exists")
    case = select_case(manifest, args.case)
    duration = case["audio"]["samples"] / case["audio"]["sample_rate"]
    verify_wav(args.audio, case["audio"])
    verify_rttm(args.rttm, case["rttm"], duration)
    go_turns, go_data = load_saved_turns(args.go_result, case["turn_extent_end"], True, expected_hash=case["go_result"]["sha256"], tie_policy=case["tie_policy"], exclusive_may_overlap=case["exclusive_may_overlap"])
    reference_turns, _ = load_saved_turns(args.reference, case["turn_extent_end"], False, expected_hash=case["reference"]["sha256"], exclusive_may_overlap=case["exclusive_may_overlap"])
    scores, reference_scores, deltas = {}, {}, {}
    passed = True
    for kind in ("full", "exclusive"):
        scores[kind] = score_turns(args.rttm, case["rttm"]["uri"], case["uem"], go_turns[kind], case["collars"])
        reference_scores[kind] = score_turns(args.rttm, case["rttm"]["uri"], case["uem"], reference_turns[kind], case["collars"])
        deltas[kind] = []
        for actual, baseline in zip(scores[kind], reference_scores[kind]):
            der_delta = 100 * (metric_value(actual, "der") - metric_value(baseline, "der"))
            jer_delta = 100 * (metric_value(actual, "jer") - metric_value(baseline, "jer"))
            deltas[kind].append({"collar": actual["collar"], "der_pp": der_delta, "jer_pp": jer_delta})
            passed &= abs(der_delta) <= case["gates"]["max_abs_der_delta_pp"]
            passed &= abs(jer_delta) <= case["gates"]["max_abs_jer_delta_pp"]
    post = go_data["Result"]["Postprocess"]
    result = {"schema": 1, "manifest_sha256": sha256(args.manifest), "case": case["id"], "role": case["role"], "audio_sha256": sha256(args.audio), "rttm_sha256": sha256(args.rttm), "go_result_sha256": sha256(args.go_result), "reference_sha256": sha256(args.reference), "uem": case["uem"], "skip_overlap": False, "tie_policy": case["tie_policy"], "training_rows": post.get("TrainingRows"), "clusters": post.get("Clusters"), "ambiguous_frames": len(post.get("Timeline", {}).get("AmbiguousFrames", [])), "scores": scores, "reference_scores": reference_scores, "deltas_pp": deltas, "gate_pass": bool(passed), "qualified": bool(passed and case["role"] == "qualifying")}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit("Community-1 saved-reference parity gate failed; evidence retained")


if __name__ == "__main__":
    main()
