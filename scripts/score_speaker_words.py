#!/usr/bin/env python3
"""Score a checked speaker transcript against independent timed speaker words.

Reports ordinary WER and concatenated permutation-invariant speaker-attributed
WER (cpSA-WER). Unlabelled hypothesis tokens are always charged as errors. This
tool does not infer reference words or speakers from system output.
"""
import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 32 << 20
MAX_WORDS = 100000
MAX_SPEAKERS = 8
POLICY = "nfkc-casefold-unicode-alnum-internal-apostrophe-v1"


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def read_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("JSON input too large")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value: {value}")))


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


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid {name}")
    return float(value)


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("invalid word text")
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    chars = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        if category[0] in ("L", "N") or char == "'" and index > 0 and index + 1 < len(text) and unicodedata.category(text[index - 1])[0] in ("L", "N") and unicodedata.category(text[index + 1])[0] in ("L", "N"):
            chars.append(char)
        else:
            chars.append(" ")
    return "".join(chars).split()


def edit_counts(reference, hypothesis):
    # Cost, substitutions, deletions, insertions; stable tie order prefers
    # correct/substitution, deletion, then insertion.
    prior = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, expected in enumerate(reference, 1):
        row = [(i, 0, i, 0)]
        for j, actual in enumerate(hypothesis, 1):
            if expected == actual:
                diagonal = prior[j - 1]
            else:
                value = prior[j - 1]
                diagonal = (value[0] + 1, value[1] + 1, value[2], value[3])
            value = prior[j]
            deletion = (value[0] + 1, value[1], value[2] + 1, value[3])
            value = row[j - 1]
            insertion = (value[0] + 1, value[1], value[2], value[3] + 1)
            row.append(min((diagonal, deletion, insertion), key=lambda item: (item[0], item[1] + item[3], item[2], item[1])))
        prior = row
    cost, substitutions, deletions, insertions = prior[-1]
    return {"errors": cost, "substitutions": substitutions, "deletions": deletions, "insertions": insertions}


def reference_words(value):
    exact_keys(value, ("schema", "corpus", "meeting", "source_interval", "sample_rate", "speakers", "words"))
    if value["schema"] != 1 or value["sample_rate"] != 16000 or not isinstance(value["words"], list) or len(value["words"]) > MAX_WORDS:
        raise ValueError("invalid reference words")
    declared = value["speakers"]
    if not isinstance(declared, list) or not 1 <= len(declared) <= MAX_SPEAKERS or len(set(declared)) != len(declared) or not all(isinstance(s, str) and s for s in declared):
        raise ValueError("invalid reference speakers")
    output = []
    prior = -1.0
    for item in value["words"]:
        exact_keys(item, ("start", "end", "speaker", "text", "source_id", "punctuation"))
        start, end = finite(item["start"], "reference start"), finite(item["end"], "reference end")
        if start < prior or start > end or item["speaker"] not in declared or not isinstance(item["source_id"], str) or not isinstance(item["punctuation"], bool):
            raise ValueError("invalid reference word")
        tokens = normalize(item["text"])
        if not tokens:
            raise ValueError("empty normalized reference word")
        output.extend((start, end, item["speaker"], token) for token in tokens)
        prior = start
    return output


def hypothesis_words(value):
    exact_keys(value, ("schema", "experimental", "transcript_key", "diarization_key", "policy", "labelled_cues", "unlabelled_cues", "labelled_words", "unlabelled_words", "transcript"))
    if value["schema"] != 2 or value["experimental"] is not True or not HEX64.fullmatch(value["transcript_key"]) or not HEX64.fullmatch(value["diarization_key"]):
        raise ValueError("invalid speaker transcript")
    transcript = value["transcript"]
    exact_keys(transcript, ("schema", "sample_rate", "total_samples", "language", "source_timing", "cues"), ("words",))
    words = transcript.get("words")
    if transcript["schema"] != 2 or transcript["sample_rate"] != 16000 or not isinstance(words, list) or len(words) > MAX_WORDS:
        raise ValueError("invalid hypothesis words")
    output = []
    prior = -1
    labelled = unlabelled = 0
    for item in words:
        exact_keys(item, ("start_sample", "end_sample", "speaker", "text"))
        start, end, speaker = item["start_sample"], item["end_sample"], item["speaker"]
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int) or start < prior or start > end or end > transcript["total_samples"] or isinstance(speaker, bool) or not isinstance(speaker, int) or not -1 <= speaker <= 63:
            raise ValueError("invalid hypothesis word")
        tokens = normalize(item["text"])
        output.extend((start / 16000, end / 16000, speaker, token) for token in tokens)
        if speaker < 0:
            unlabelled += 1
        else:
            labelled += 1
        prior = start
    if labelled != value["labelled_words"] or unlabelled != value["unlabelled_words"]:
        raise ValueError("speaker transcript accounting changed")
    return output


def assignment_cost(reference, hypothesis):
    ref_labels = sorted({word[2] for word in reference})
    hyp_labels = sorted({word[2] for word in hypothesis if word[2] >= 0})
    if len(ref_labels) > MAX_SPEAKERS or len(hyp_labels) > MAX_SPEAKERS:
        raise ValueError("too many speakers")
    ref_streams = {label: [word[3] for word in reference if word[2] == label] for label in ref_labels}
    hyp_streams = {label: [word[3] for word in hypothesis if word[2] == label] for label in hyp_labels}
    size = max(len(ref_labels), len(hyp_labels))
    refs = ref_labels + [None] * (size - len(ref_labels))
    hyps = hyp_labels + [None] * (size - len(hyp_labels))
    costs = [[edit_counts(ref_streams.get(ref, []), hyp_streams.get(hyp, []))["errors"] for hyp in hyps] for ref in refs]
    states = {0: (0, [])}
    for row in range(size):
        following = {}
        for mask, (cost, mapping) in states.items():
            for column in range(size):
                if mask & (1 << column):
                    continue
                candidate = (cost + costs[row][column], mapping + [column])
                key = mask | (1 << column)
                if key not in following or candidate < following[key]:
                    following[key] = candidate
        states = following
    cost, columns = states[(1 << size) - 1]
    mapping = [{"reference": refs[row], "hypothesis": hyps[column], "errors": costs[row][column]} for row, column in enumerate(columns)]
    return cost, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-words", required=True)
    parser.add_argument("--speaker-transcript", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("output already exists")
    reference = reference_words(read_json(args.reference_words))
    hypothesis = hypothesis_words(read_json(args.speaker_transcript))
    lexical = edit_counts([word[3] for word in reference], [word[3] for word in hypothesis])
    stream_errors, mapping = assignment_cost(reference, hypothesis)
    unlabelled = sum(word[2] < 0 for word in hypothesis)
    speaker_errors = stream_errors + unlabelled
    denominator = len(reference)
    result = {
        "schema": 1,
        "normalization": POLICY,
        "reference_words_sha256": sha256(args.reference_words),
        "speaker_transcript_sha256": sha256(args.speaker_transcript),
        "reference_words": denominator,
        "hypothesis_words": len(hypothesis),
        "reference_speakers": len({word[2] for word in reference}),
        "hypothesis_speakers": len({word[2] for word in hypothesis if word[2] >= 0}),
        "unlabelled_hypothesis_words": unlabelled,
        "wer": {**lexical, "rate": lexical["errors"] / denominator},
        "cp_sawer": {"errors": speaker_errors, "stream_errors": stream_errors, "unlabelled_errors": unlabelled, "rate": speaker_errors / denominator, "mapping": mapping},
        "qualified": False,
        "scope": "absolute pilot metrics only; no ratified corpus budget or pinned reference-system delta",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
