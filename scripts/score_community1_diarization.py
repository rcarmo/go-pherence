#!/usr/bin/env python3
"""Score saved Go turns against pinned public RTTM; no neural inference.

Reports full/exclusive DER with explicit30s UEM, overlap included and0/0.25s
collars. An optional saved pyannote baseline is comparison evidence, not a new
reference run or proof of matching neural configuration. No tolerance widening.
"""
import argparse
import hashlib
import json
from pathlib import Path

RTTM_SHA = "d78fe62c69d8e6dcbb42c26adfce83faccb374c5a1e6d987fe37f85f1c173c87"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--go-result", required=True)
    p.add_argument("--rttm", required=True)
    p.add_argument("--baseline", help="optional saved diar-result.json from pyannote reference")
    p.add_argument("--reference-dir", help="fresh pinned pipeline oracle directory with JSON/NPY")
    p.add_argument("--output", required=True)
    p.add_argument("--require-reference-parity", action="store_true", help="fail after saving evidence unless fresh masks/embeddings/turns match")
    args = p.parse_args()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(args.rttm) != RTTM_SHA:
        raise ValueError("reference RTTM hash changed")
    if Path(args.output).exists():
        raise ValueError("output already exists")
    from pyannote.database.util import load_rttm
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    reference = load_rttm(args.rttm)["sample"]
    uem = Timeline([Segment(0, 30)], uri="sample")
    data = json.loads(Path(args.go_result).read_text())
    result = data["Result"]
    if len(result["Windows"]) != 21 or result["Grid"]["Frames"] != 589:
        raise ValueError("expected pinned30s/10s-window/1s-step result")
    def score(turns, go):
        annotation = Annotation(uri="sample")
        for i, turn in enumerate(turns):
            start, end, speaker = (turn["Start"], turn["End"], turn["Speaker"]) if go else (turn["start"], turn["end"], turn["speaker"])
            if not 0 <= start < end <= 40:
                raise ValueError("invalid turn bounds")
            annotation[Segment(start, end), i] = str(speaker)
        scores = []
        for collar in (0., .25):
            metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
            detail = metric(reference, annotation, uem=uem, detailed=True)
            scores.append(dict(collar=collar, details={k: float(v) for k, v in detail.items()}))
        return scores
    scores = {kind: score(result["Postprocess"][key], True) for kind, key in [("full", "FullTurns"), ("exclusive", "ExclusiveTurns")]}
    baseline = None
    if args.baseline:
        b = json.loads(Path(args.baseline).read_text())
        if b["model_revision"] != "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee":
            raise ValueError("baseline model revision changed")
        matched = [r for r in b["results"] if r["variant"] == "wav" and r["backend"] == "ff"]
        if len(matched) != 1 or matched[0]["evaluated_frames"] != 480000:
            raise ValueError("baseline common interval changed")
        ref_scores = score(matched[0]["turns"], False)
        baseline = dict(sha256=sha(args.baseline), scope="saved historical pyannote WAV/FF frontend reference, not a fresh matched-source neural run", scores=ref_scores,
                        full_der_delta_pp=[100*(a["details"]["diarization error rate"]-b["details"]["diarization error rate"]) for a, b in zip(scores["full"], ref_scores)])
    fresh = None
    if args.reference_dir:
        import itertools
        import numpy as np
        directory = Path(args.reference_dir)
        reference_data = json.loads((directory/"reference.json").read_text())
        if reference_data["model_revision"] != "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee":
            raise ValueError("fresh reference model changed")
        for name, info in reference_data["artifacts"].items():
            if sha(directory/(name+".npy")) != info["sha256"]:
                raise ValueError("reference artifact checksum")
        seg = np.load(directory/"segmentation.npy", allow_pickle=False)
        emb = np.load(directory/"embeddings.npy", allow_pickle=False)
        if seg.shape != (21,589,3) or emb.shape != (21,3,256):
            raise ValueError("fresh reference tensor shapes")
        go_seg = np.asarray(result["Segmentations"], dtype=np.float32).reshape(seg.shape)
        go_emb = np.asarray(result["Embeddings"], dtype=np.float32).reshape(emb.shape)
        if not np.isfinite(emb).all() or not np.isfinite(go_emb).all():
            raise ValueError("nonfinite comparison embeddings")
        labels = sorted({t["speaker"] for t in reference_data["full"]})
        clusters = result["Postprocess"]["Clusters"]
        if clusters != len(labels) or clusters > 8:
            raise ValueError("unsupported label permutation comparison")
        mappings=[]
        for permutation in itertools.permutations(labels):
            counts=[]
            for kind,key in [("full","FullTurns"),("exclusive","ExclusiveTurns")]:
                a=sorted((v["Start"],v["End"],permutation[v["Speaker"]]) for v in result["Postprocess"][key])
                b=sorted((v["start"],v["end"],v["speaker"]) for v in reference_data[kind])
                counts.append(len(a)==len(b) and all(abs(x[0]-y[0])<=1e-12 and abs(x[1]-y[1])<=1e-12 and x[2]==y[2] for x,y in zip(a,b)))
            if all(counts): mappings.append(list(permutation))
        fresh_scores={kind:score(reference_data[kind],False) for kind in ("full","exclusive")}
        fresh=dict(reference_sha256=sha(directory/"reference.json"),scores=fresh_scores,
                   full_der_delta_pp=[100*(a["details"]["diarization error rate"]-b["details"]["diarization error rate"]) for a,b in zip(scores["full"],fresh_scores["full"])],
                   exact_segmentation_values=int((seg==go_seg).sum()),segmentation_values=int(seg.size),
                   embeddings_max_abs=float(np.max(np.abs(emb.astype(np.float64)-go_emb.astype(np.float64)))),
                   embeddings_values=int(emb.size), matching_turn_label_permutations=mappings,
                   turn_boundary_tolerance_seconds=1e-12)
    output = dict(schema=1, go_result_sha256=sha(args.go_result), rttm_sha256=RTTM_SHA, uem=[0,30], skip_overlap=False,
                  config=data["Config"], training_rows=result["Postprocess"]["TrainingRows"], clusters=result["Postprocess"]["Clusters"],
                  ambiguous_frames=len(result["Postprocess"]["Timeline"]["AmbiguousFrames"]), scores=scores, historical_baseline=baseline, fresh_reference=fresh,
                  qualified=False, scope="one public sample, explicit lowest-index tie policy, strict neural intermediates still fail; no general DER or performance qualification")
    Path(args.output).write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output))
    if args.require_reference_parity:
        if (fresh is None or fresh["exact_segmentation_values"] != fresh["segmentation_values"]
                or fresh["embeddings_max_abs"] > 2e-4 or not fresh["matching_turn_label_permutations"]
                or any(abs(delta) > 1e-9 for delta in fresh["full_der_delta_pp"])):
            raise SystemExit("fresh reference parity gate failed; evidence retained")


if __name__ == "__main__":
    main()
