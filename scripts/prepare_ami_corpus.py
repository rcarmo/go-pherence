#!/usr/bin/env python3
"""Prepare one hash-pinned AMI meeting excerpt for speech qualification.

Inputs are the official AMI headset-mix WAV and manual-annotation ZIP. The
script downloads nothing, trusts no archive paths, and refuses to overwrite its
output directory. It emits canonical 16 kHz mono PCM, clipped RTTM, and timed
speaker-word JSON. Source media belongs in ignored local storage, not Git.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import wave
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MEETING_RE = re.compile(r"^[A-Z]{2}[0-9]{4}[a-z]$")
SPEAKERS = "ABCD"
NITE_ID = "{http://nite.sourceforge.net/}id"
MAX_ARCHIVE_BYTES = 64 << 20
MAX_MEMBER_BYTES = 8 << 20


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(path, expected, label):
    if not SHA256_RE.fullmatch(expected):
        raise ValueError(f"invalid {label} SHA-256")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 changed: {actual}")


def archive_member(archive, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe archive member")
    info = archive.getinfo(name)
    if info.is_dir() or info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"invalid archive member size: {name}")
    return archive.read(info)


def finite_time(value, label):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"invalid {label}")
    return parsed


def parse_annotations(annotation_zip, meeting):
    turns, words = [], []
    with zipfile.ZipFile(annotation_zip) as archive:
        for speaker in SPEAKERS:
            segment_name = f"segments/{meeting}.{speaker}.segments.xml"
            word_name = f"words/{meeting}.{speaker}.words.xml"
            segment_root = ET.fromstring(archive_member(archive, segment_name))
            word_root = ET.fromstring(archive_member(archive, word_name))
            expected_roots = (f"{meeting}.{speaker}.segs", f"{meeting}.{speaker}.words")
            roots = (segment_root, word_root)
            if any(root.tag != "{http://nite.sourceforge.net/}root" for root in roots) or tuple(root.attrib.get(NITE_ID) for root in roots) != expected_roots:
                raise ValueError("unexpected AMI annotation root")
            for element in segment_root:
                if not element.tag.endswith("segment"):
                    raise ValueError("unexpected segment element")
                start = finite_time(element.attrib.get("transcriber_start"), "segment start")
                end = finite_time(element.attrib.get("transcriber_end"), "segment end")
                if start >= end:
                    raise ValueError("nonpositive segment")
                turns.append((start, end, speaker))
            for element in word_root:
                if not element.tag.endswith("w"):
                    continue
                start = finite_time(element.attrib.get("starttime"), "word start")
                end = finite_time(element.attrib.get("endtime"), "word end")
                text = "".join(element.itertext()).strip()
                word_id = element.attrib.get(NITE_ID, "")
                if not word_id or start > end or not text:
                    raise ValueError("invalid timed word")
                words.append((start, end, speaker, text, word_id, element.attrib.get("punc") == "true"))
    turns.sort()
    words.sort()
    return turns, words


def clip_interval(start, end, clip_start, clip_end):
    start, end = max(start, clip_start), min(end, clip_end)
    if start >= end:
        return None
    return start - clip_start, end - clip_start


def write_wav(source, output, start_sample, sample_count):
    with wave.open(str(source), "rb") as reader:
        geometry = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype())
        if geometry != (1, 2, 16000, "NONE"):
            raise ValueError(f"unsupported AMI WAV geometry: {geometry}")
        if start_sample + sample_count > reader.getnframes():
            raise ValueError("excerpt exceeds source WAV")
        reader.setpos(start_sample)
        frames = reader.readframes(sample_count)
        if len(frames) != sample_count * 2:
            raise ValueError("short WAV read")
        with wave.open(str(output), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--annotations-sha256", required=True)
    parser.add_argument("--start", type=int, required=True, help="whole-second excerpt start")
    parser.add_argument("--duration", type=int, required=True, help="whole-second excerpt duration")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not MEETING_RE.fullmatch(args.meeting):
        raise ValueError("invalid AMI meeting id")
    if args.start < 0 or not 1 <= args.duration <= 3600:
        raise ValueError("invalid excerpt geometry")
    audio, annotations, output = Path(args.audio), Path(args.annotations), Path(args.output)
    if annotations.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("annotation archive too large")
    checked_hash(audio, args.audio_sha256, "audio")
    checked_hash(annotations, args.annotations_sha256, "annotations")
    if output.exists():
        raise ValueError("output already exists")
    output.mkdir(parents=True)
    try:
        clip_start, clip_end = float(args.start), float(args.start + args.duration)
        turns, words = parse_annotations(annotations, args.meeting)
        clipped_turns = []
        for start, end, speaker in turns:
            clipped = clip_interval(start, end, clip_start, clip_end)
            if clipped:
                clipped_turns.append((*clipped, speaker))
        clipped_words = []
        for start, end, speaker, text, word_id, punctuation in words:
            clipped = clip_interval(start, end, clip_start, clip_end)
            if clipped:
                clipped_words.append({"start": clipped[0], "end": clipped[1], "speaker": speaker, "text": text, "source_id": word_id, "punctuation": punctuation})
        if not clipped_turns or not clipped_words or set(turn[2] for turn in clipped_turns) != set(SPEAKERS):
            raise ValueError("excerpt lacks complete speaker/word coverage")
        wav_path = output / f"{args.meeting}-{args.start}-{args.start + args.duration}.wav"
        rttm_path = output / f"{args.meeting}-{args.start}-{args.start + args.duration}.rttm"
        words_path = output / f"{args.meeting}-{args.start}-{args.start + args.duration}.words.json"
        write_wav(audio, wav_path, args.start * 16000, args.duration * 16000)
        uri = f"{args.meeting}-{args.start}-{args.start + args.duration}"
        with open(rttm_path, "w", encoding="utf-8", newline="\n") as stream:
            for start, end, speaker in clipped_turns:
                stream.write(f"SPEAKER {uri} 1 {start:.6f} {end-start:.6f} <NA> <NA> {speaker} <NA> <NA>\n")
        words_path.write_text(json.dumps({"schema": 1, "corpus": "AMI Meeting Corpus", "meeting": args.meeting, "source_interval": [args.start, args.start + args.duration], "sample_rate": 16000, "speakers": list(SPEAKERS), "words": clipped_words}, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {"schema": 1, "meeting": args.meeting, "source_interval": [args.start, args.start + args.duration], "source": {"audio_sha256": sha256(audio), "annotations_sha256": sha256(annotations)}, "output": {"audio": {"path": wav_path.name, "sha256": sha256(wav_path), "samples": args.duration * 16000}, "rttm": {"path": rttm_path.name, "sha256": sha256(rttm_path), "turns": len(clipped_turns), "uri": uri}, "words": {"path": words_path.name, "sha256": sha256(words_path), "count": len(clipped_words)}}, "licence": {"spdx": "CC-BY-4.0", "url": "https://groups.inf.ed.ac.uk/ami/corpus/license.shtml"}}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, sort_keys=True))
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
