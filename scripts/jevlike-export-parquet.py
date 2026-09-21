#!/usr/bin/env python3
"""Export a pinned 4,800-original Jevlike pilot; no network or training.

Conversion-only dependency: pyarrow==21.0.0. Go owns task adaptation, candidate
construction, deduplication and the final train/validation/calibration splits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def select_groups(rows, source, split, count, seed):
    """Training selection uses only group IDs and label/genre for stratification."""
    buckets = {}
    rejected = 0
    exclusions = []
    for index, row in enumerate(rows):
        if source == "multi_nli" and (row["genre"] == "fiction" or row["label"] not in (0, 1, 2)):
            rejected += 1
            continue
        if source in ("commonsense_qa", "ai2_arc"):
            choices = row["choices"]["text"]
            labels = row["choices"]["label"]
            if not 2 <= len(choices) <= 32 or len(set(choices)) != len(choices) or len(set(labels)) != len(labels) or row["answerKey"] not in labels or any(not c.strip() for c in choices):
                exclusions.append({"source_id": row["id"], "reason": "ambiguous_or_invalid_single_choice"})
                continue
        if source == "multi_nli":
            group = str(row["promptID"])
            stratum = row["genre"]  # label-stratifying rows would break prompt groups
        elif source == "clinc_oos":
            group = hashlib.sha256(row["text"].encode()).hexdigest()
            stratum = str(row["intent"])
            row = dict(row, id=group)
        else:
            group = row["id"]
            stratum = "all"  # factual QA alternatives have no global label class
        buckets.setdefault(stratum, {}).setdefault(group, []).append(row)
    ordered = {}
    for stratum, groups in buckets.items():
        ordered[stratum] = [(group, groups[group]) for group in sorted(groups, key=lambda g: rank(seed, source, split, stratum, g))]
    selected = []
    cursors = {key: 0 for key in ordered}
    # Round-robin strata, never split a source group to hit an exact quota.
    while len(selected) < count:
        progressed = False
        for stratum in sorted(ordered):
            cursor = cursors[stratum]
            if cursor >= len(ordered[stratum]):
                continue
            cursors[stratum] += 1
            group, items = ordered[stratum][cursor]
            progressed = True
            if len(selected) + len(items) <= count:
                selected.extend(items)
        if not progressed:
            break
    selected.sort(key=lambda row: rank(seed, source, split, json.dumps(row, sort_keys=True, ensure_ascii=False)))
    return selected, {"input_rows": len(rows), "excluded_fiction_or_unlabelled": rejected, "selected": len(selected), "quota": count, "strata": len(ordered), "excluded_invalid_choices": exclusions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints/jevlike-qwen3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    repo = Path(__file__).resolve().parents[1]
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("output must not exist")
    if output == repo or repo / "model" in output.parents:
        raise SystemExit("dataset exports must not be written into source")
    pins = json.loads((repo / "docs/experiments/jevlike-qwen3/sources.json").read_text())
    # Explicit model-independent quotas; final tests are not used for selection.
    plan = {
        "multi_nli": [("", "data/train-00000-of-00001.parquet", "train", 1200),
                      ("", "data/validation_matched-00000-of-00001.parquet", "validation_matched", 160),
                      ("", "data/validation_mismatched-00000-of-00001.parquet", "validation_mismatched", 160)],
        "commonsense_qa": [("default", "data/train-00000-of-00001.parquet", "train", 1200),
                           ("default", "data/validation-00000-of-00001.parquet", "validation", 160)],
        "ai2_arc": [(config, f"{config}/{split}-00000-of-00001.parquet", split, 600 if split == "train" else 160)
                    for config in ["ARC-Easy", "ARC-Challenge"] for split in ["train", "validation", "test"]],
        "clinc_oos": [("plus", f"plus/{split}-00000-of-00001.parquet", split, 1200 if split == "train" else 160)
                      for split in ["train", "validation", "test"]],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".jevlike-export-", dir=output.parent))
    try:
        inputs, selections = [], []
        for source in pins["datasets"]:
            directory = root / source["repository"].replace("/", "--") / source["revision"]
            files = {entry["path"]: entry for entry in source["files"]}
            for config, file, split, quota in plan[source["id"]]:
                path = directory / file
                expected = files[file]
                if path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
                    raise ValueError(f"source identity mismatch: {path}")
                parquet = pq.ParquetFile(path)
                rows = parquet.read().to_pylist()
                selected, stats = select_groups(rows, source["id"], split, quota, args.seed)
                name = "-".join([source["id"], config or "default", split]) + ".jsonl"
                dest = temporary / name
                with dest.open("w", encoding="utf8") as f:
                    for row in selected:
                        if source["id"] == "multi_nli":
                            # The pinned source reuses pairID for distinct hypotheses.
                            # Preserve it and add an exact-content disambiguator;
                            # promptID remains the grouping boundary.
                            row = dict(row)
                            row["original_pairID"] = row["pairID"]
                            row["pairID"] += ":" + rank(0, row["premise"], row["hypothesis"])[:16]
                        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                entry = {"source": source["id"], "config": config, "revision": source["revision"],
                         "license": ",".join(source["license"]), "split": split, "path": name, "sha256": digest(dest)}
                if source["id"] == "clinc_oos":
                    names = json.loads(parquet.schema_arrow.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
                    if len(names) != 151 or names.count("oos") != 1:
                        raise ValueError("unexpected pinned CLINC intent vocabulary")
                    entry.update(labels=["out of scope intent" if n == "oos" else n.replace("_", " ") for n in names],
                                 candidate_count=8, candidate_seed=args.seed, oos_id=names.index("oos"))
                inputs.append(entry)
                selections.append(dict(source=source["id"], config=config, split=split, parquet_sha256=expected["sha256"], **stats))
        (temporary / "inputs.json").write_text(json.dumps({"version": 1, "inputs": inputs}, indent=2) + "\n")
        (temporary / "selection.json").write_text(json.dumps({"version": 1, "seed": args.seed,
            "policy": "source-group hash and round-robin genre/intent strata; no augmentation; fiction excluded",
            "partitions": selections}, indent=2) + "\n")
        if output.exists():
            raise ValueError("output appeared during export")
        os.rename(temporary, output)
        print(json.dumps({"output": str(output), "inputs": len(inputs), "rows": sum(s["selected"] for s in selections)}))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
