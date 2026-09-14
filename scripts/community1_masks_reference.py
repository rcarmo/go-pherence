#!/usr/bin/env python3
"""Extract pinned pyannote mask-selection/filter methods for synthetic fixtures.

CPU one thread; no checkpoints, inference, media, pipeline imports, or GPU.
get_embeddings runs with an embedding stub returning the mask unchanged. The
real method still performs clean-mask choice, NaN conversion and batch reshape.
"""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

DIAR_SHA = "cbb358abedef5042fcc71bb970b12a2936a16686be4bda15691a224f172656fd"
CLUSTER_SHA = "6031fb7c21277a7e9901ef2cdaed7d5cd69f7ef45508dc4b45e82ce0da3c8fba"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diarization", required=True)
    parser.add_argument("--clustering", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path, expected in [(args.diarization, DIAR_SHA), (args.clustering, CLUSTER_SHA)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise SystemExit("source checksum mismatch")
    import numpy as np
    import torch
    from einops import rearrange
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")

    class Window:
        def __init__(self, data, sliding_window):
            self.data, self.sliding_window = data, sliding_window

        def __iter__(self):
            return iter(enumerate(self.data))

    class Audio:
        def __init__(self, samples):
            self.samples = samples

        def crop(self, file, chunk, mode):
            assert mode == "pad"
            return torch.zeros((1, self.samples)), 16000

    class Embedding:
        sample_rate = 16000

        def __init__(self, minimum):
            self.min_num_samples = minimum

        def __call__(self, waveforms, masks):
            return masks.numpy().copy()

    def batchify(items, batch_size, fillvalue):
        return itertools.zip_longest(*[iter(items)] * batch_size, fillvalue=fillvalue)

    scope = dict(np=np, torch=torch, math=math, rearrange=rearrange,
                 SlidingWindowFeature=Window, batchify=batchify)

    def extract(path, cls_name, method):
        tree = ast.parse(Path(path).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, path, "exec"), scope)
        return scope[method]

    choose = extract(args.diarization, "SpeakerDiarization", "get_embeddings")
    admit = extract(args.clustering, "BaseClustering", "filter_embeddings")
    selections, filters = [], []

    def array(values):
        # JSON null encodes synthetic NaN input only. Go fixtures restore it.
        return [None if math.isnan(float(x)) else float(x) for x in np.asarray(values).flatten()]

    selection_inputs = [
        ("strict_equal_fallback", [[1, 0], [1, 0], [1, 1], [1, 1], [0, 0]], 100, 40),
        ("strict_above_clean", [[1, 0], [1, 0], [1, 0], [1, 1], [0, 0]], 100, 40),
        ("overlap_only", [[1, 1]] * 7, 160000, 400),
        ("empty", [[0, 0, 0]] * 9, 160000, 400),
        ("partial_unknown", [[1, 0], [1, np.nan], [1, 0], [0, 1], [np.nan, 1]], 100, 1),
        ("above_window", [[1, 0], [0, 1], [1, 0]], 80, 200),
        ("ceil_fraction", [[1, 0], [1, 1], [0, 1], [1, 0], [0, 1]], 71, 15),
    ]
    rng = np.random.default_rng(7301)
    for i in range(8):
        frames, speakers = 7 + 3 * i, 1 + i % 4
        data = rng.integers(0, 2, (frames, speakers)).astype(np.float32)
        if i % 2:
            data[i % frames, i % speakers] = np.nan
        selection_inputs.append((f"seeded_{i}", data, 1600, 31 + i * 47))
    for name, data, samples, minimum in selection_inputs:
        data = np.asarray(data, dtype=np.float32)
        frames, speakers = data.shape
        for exclude in [False, True]:
            owner = SimpleNamespace(training=False, _embedding=Embedding(minimum), _audio=Audio(samples), embedding_batch_size=3)
            masks = choose(owner, {}, Window(data[None], SimpleNamespace(duration=samples / 16000)), exclude_overlap=exclude)
            assert masks.shape == (1, speakers, frames)
            selections.append(dict(name=f"{name}_{exclude}", config=dict(Frames=frames, Speakers=speakers, WindowSamples=samples, MinimumSamples=minimum, ExcludeOverlap=exclude),
                                   segmentations=array(data), masks=array(masks), selected_frames=np.sum(masks[0], axis=1).astype(int).tolist(),
                                   minimum_clean_frames=math.ceil(frames * minimum / (samples / 16000 * 16000)) if exclude else -1))

    data = np.array([[[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]],
                     [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.float32)
    for variant in ["finite", "nan_embedding", "nan_segmentation", "overlap_only", "empty"]:
        seg = data.copy()
        embeddings = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) * .125
        if variant == "nan_embedding":
            embeddings[0, 1, 2] = np.nan
        elif variant == "nan_segmentation":
            seg[0, 3, 0] = np.nan
            seg[1, 1, 1] = np.nan
        elif variant == "overlap_only":
            seg[:] = 1
        elif variant == "empty":
            seg[:] = 0
        # NumPy weak scalar promotion casts this just-above-.2 threshold to
        # float32 before comparing with float32 clean counts (still admits 1/5).
        for ratio in [0., .2, .4, 1., float(np.nextafter(.2, 1.))]:
            filtered, chunks, speakers = admit(None, embeddings, Window(seg, None), min_active_ratio=ratio)
            filters.append(dict(name=f"{variant}_{ratio}", config=dict(Chunks=2, Frames=5, Speakers=3, Dimension=4, MinActiveRatio=ratio),
                                segmentations=array(seg), embeddings=array(embeddings), filtered=array(filtered),
                                chunk_indices=chunks.tolist(), speaker_indices=speakers.tolist()))
    result = dict(schema=1, reference=dict(pyannote_commit="b749285c5cdd4636b2edc7f766f1352c8dde9369", diarization_sha256=DIAR_SHA, clustering_sha256=CLUSTER_SHA, numpy=np.__version__, torch=torch.__version__, cpu_threads=1),
                  scope="actual extracted methods, synthetic binary/NaN masks and embedding stub; no trained inference", selections=selections, filters=filters)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {len(selections)} mask selections and {len(filters)} clustering filters")


if __name__ == "__main__":
    main()
