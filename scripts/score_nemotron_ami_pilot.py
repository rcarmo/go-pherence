#!/usr/bin/env python3
"""Score saved Nemotron CLI text and speaker spans on pinned AMI excerpts.

This scores absolute transcript-only WER and overlap-inclusive DER/JER. CLI
ASR has no word times or speaker labels, so cpSA-WER cannot be measured here.
No inference, downloading, or corpus-level quality decision occurs here.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent
# Excerpt geometry is selected before reading the locally prepared manifest.
# Source and generated-asset hashes are separately pinned by each committed
# contract file and must match the local manifest and actual assets.
CASE_GEOMETRY = {(350, 410): (28, 179), (350, 450): (39, 237), (530, 630): (36, 269)}


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORPUS = load_module("score_community1_corpus")
WORDS = load_module("score_speaker_words")


def score(contract_path, manifest_path, audio_path, rttm_path, words_path, transcript_path, turns_path):
    contract = CORPUS.read_json(contract_path)
    manifest = CORPUS.read_json(manifest_path)
    CORPUS.exact_keys(manifest, ("schema", "meeting", "source_interval", "source", "output", "licence"))
    if manifest != contract:
        raise ValueError("AMI pilot manifest differs from pinned contract")
    interval = manifest["source_interval"]
    if manifest["schema"] != 1 or manifest["meeting"] != "ES2004a" or not isinstance(interval, list) or len(interval) != 2 or any(type(v) is not int for v in interval) or tuple(interval) not in CASE_GEOMETRY:
        raise ValueError("unsupported AMI pilot identity")
    start, end = interval
    duration = end - start
    expected_turns, expected_words = CASE_GEOMETRY[start, end]
    uri = f"ES2004a-{start}-{end}"
    outputs = manifest["output"]
    CORPUS.exact_keys(outputs, ("audio", "rttm", "words"))
    audio, rttm, words = outputs["audio"], outputs["rttm"], outputs["words"]
    CORPUS.exact_keys(audio, ("path", "sha256", "samples"))
    CORPUS.exact_keys(rttm, ("path", "sha256", "turns", "uri"))
    CORPUS.exact_keys(words, ("path", "sha256", "count"))
    if (audio["path"], rttm["path"], words["path"], rttm["uri"]) != (
        uri + ".wav", uri + ".rttm", uri + ".words.json", uri
    ) or audio["samples"] != duration * 16000 or rttm["turns"] != expected_turns or words["count"] != expected_words:
        raise ValueError("unexpected AMI pilot geometry")
    for path, expected in ((audio_path, audio["sha256"]), (rttm_path, rttm["sha256"]), (words_path, words["sha256"])):
        if not isinstance(expected, str) or not CORPUS.HEX64.fullmatch(expected) or CORPUS.sha256(path) != expected:
            raise ValueError("AMI pilot asset checksum changed")
    CORPUS.verify_wav(audio_path, {"sha256": audio["sha256"], "samples": audio["samples"], "channels": 1, "sample_rate": 16000})
    CORPUS.verify_rttm(rttm_path, rttm, duration)
    reference = WORDS.read_json(words_path)
    if reference["meeting"] != manifest["meeting"] or reference["source_interval"] != manifest["source_interval"] or len(reference["words"]) != words["count"]:
        raise ValueError("AMI pilot word identity changed")
    reference_tokens = WORDS.reference_words(reference)
    if Path(transcript_path).stat().st_size > WORDS.MAX_BYTES:
        raise ValueError("transcript too large")
    hypothesis_tokens = WORDS.normalize(Path(transcript_path).read_text(encoding="utf-8"))
    if not hypothesis_tokens or len(hypothesis_tokens) > WORDS.MAX_WORDS:
        raise ValueError("invalid transcript tokens")
    lexical = WORDS.edit_counts([word[3] for word in reference_tokens], hypothesis_tokens)
    turns = CORPUS.parse_turns(CORPUS.read_json(turns_path), duration, True)
    if not turns or any(int(speaker) > 7 for _, _, speaker in turns):
        raise ValueError("invalid Nemotron speaker spans")
    scores = CORPUS.score_turns(rttm_path, rttm["uri"], [[0, duration]], turns, [0.0, 0.25])
    return {
        "schema": 1,
        "meeting": manifest["meeting"],
        "source_interval": manifest["source_interval"],
        "audio_samples": audio["samples"],
        "inputs_sha256": {
            "contract": CORPUS.sha256(contract_path),
            "manifest": CORPUS.sha256(manifest_path),
            "audio": audio["sha256"],
            "rttm": rttm["sha256"],
            "words": words["sha256"],
            "transcript": CORPUS.sha256(transcript_path),
            "spans": CORPUS.sha256(turns_path),
        },
        "normalization": WORDS.POLICY,
        "wer": {"reference_tokens": len(reference_tokens), "hypothesis_tokens": len(hypothesis_tokens), **lexical, "rate": lexical["errors"] / len(reference_tokens)},
        "diarization": {"hypothesis_spans": len(turns), "uem": [0, duration], "skip_overlap": False, "scores": scores},
        "qualified": False,
        "scope": "one AMI headset-mix excerpt; absolute transcript-only WER and DER/JER; no word timestamps, cpSA-WER, corpus gate or reference-system delta",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "manifest", "audio", "rttm", "words", "transcript", "spans", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("output already exists")
    result = score(args.contract, args.manifest, args.audio, args.rttm, args.words, args.transcript, args.spans)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
