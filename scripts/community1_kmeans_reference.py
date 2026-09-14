#!/usr/bin/env python3
"""Generate model-free Community-1 forced-count KMeans fixtures.

Uses pinned NumPy/scikit-learn only. No model, audio, network or GPU work.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.random.mtrand as mtrand
import sklearn
from sklearn.cluster import KMeans
import sklearn.cluster._kmeans as km
import sklearn.cluster._k_means_common as km_common
import sklearn.cluster._k_means_lloyd as km_lloyd

NUMPY_VERSION = "2.5.3"
SKLEARN_VERSION = "1.9.0"
MTRAND_SHA = "53a0dea25039fd6e55ce2a6c94789c9273fdfa1cc4bda60dde6e813f455c9bc5"
KMEANS_SHA = "7d9cd3c75f1c40616223fbceb23bc1e115de043b355ab90716898301756d746c"
KMEANS_COMMON_SHA = "28273c46e0c190277ad7739aadc086c7dacf6a9ad9e0232a26f223b67274b25e"
KMEANS_LLOYD_SHA = "888274137c2f633b5814926ebdea181b7c508222549d0ad7e0c5c4554e831214"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    expected = {
        "NumPy version": (np.__version__, NUMPY_VERSION),
        "scikit-learn version": (sklearn.__version__, SKLEARN_VERSION),
        "NumPy mtrand": (sha(mtrand.__file__), MTRAND_SHA),
        "scikit-learn KMeans": (sha(km.__file__), KMEANS_SHA),
        "scikit-learn common kernel": (sha(km_common.__file__), KMEANS_COMMON_SHA),
        "scikit-learn Lloyd kernel": (sha(km_lloyd.__file__), KMEANS_LLOYD_SHA),
    }
    for name, (got, want) in expected.items():
        if got != want:
            raise SystemExit(f"{name} mismatch: got {got}, want {want}")
    cases = [
        ("two_blobs", 2, [[1.0, .1, 0], [.9, -.1, .05], [1.1, .05, -.04], [-1, .1, 0], [-.9, -.2, .03], [-1.1, 0, -.08]]),
        ("three_blobs", 3, [[1, .1, 0, 0], [.8, -.1, .1, 0], [1.2, .05, -.1, 0], [0, 1, .1, 0], [.1, .9, -.1, .05], [-.1, 1.1, 0, -.05], [0, 0, 1, .1], [.05, -.1, .9, 0], [-.05, .1, 1.1, 0]]),
        ("uneven_four", 4, [[1, 0], [.98, .03], [1.03, -.02], [0, 1], [.02, .96], [-1, 0], [-.95, -.04], [0, -1], [.03, -.9], [-.02, -1.1]]),
        ("close_clusters", 3, [[1, .01, 0], [.99, .02, 0], [.96, .28, .02], [.94, .31, -.01], [-1, 0, .01], [-.98, -.02, -.01], [-.92, -.35, .02], [-.9, -.39, -.02]]),
        ("single_cluster", 1, [[1, 0, .1], [.8, .2, 0], [-.3, .9, .1], [-1, -.1, .2]]),
        ("duplicates", 3, [[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [-1, 0], [.01, .99]]),
    ]
    seeded = np.random.RandomState(9917).normal(size=(31, 7)).astype(np.float32)
    cases.append(("seeded_irregular", 5, seeded.tolist()))
    matrix_rng = np.random.RandomState(20260912)
    for index in range(20):
        rows = 8 + index * 2
        dimension = 2 + index % 11
        clusters = 2 + index % min(6, rows - 1)
        values = matrix_rng.normal(size=(rows, dimension)).astype(np.float32)
        # Deterministic anisotropy/offsets exercise mean subtraction and greedy seeding.
        values *= np.linspace(.25, 2.0, dimension, dtype=np.float32)
        values += np.float32((index % 5) - 2) * .125
        cases.append((f"seeded_matrix_{index:02d}", clusters, values.tolist()))
    out = []
    for name, clusters, values in cases:
        original = np.asarray(values, dtype=np.float32)
        normalized = original / np.linalg.norm(original, axis=1, keepdims=True)
        labels = KMeans(n_clusters=clusters, n_init=3, random_state=42, copy_x=False).fit_predict(normalized.copy())
        centroids = np.vstack([np.mean(original[labels == k], axis=0) for k in range(clusters)])
        out.append({"name": name, "rows": original.shape[0], "dimension": original.shape[1], "clusters": clusters,
                    "input": original.flatten().tolist(), "normalized": normalized.flatten().tolist(),
                    "labels": labels.tolist(), "centroids": centroids.flatten().tolist()})
    rng = np.random.RandomState(42)
    result = {"schema": 1, "reference": {"numpy": np.__version__, "sklearn": sklearn.__version__,
               "numpy_mtrand_sha256": sha(mtrand.__file__), "kmeans_py_sha256": sha(km.__file__),
               "kmeans_common_sha256": sha(km_common.__file__), "kmeans_lloyd_sha256": sha(km_lloyd.__file__),
               "random_state": 42, "n_init": 3,
               "scope": "synthetic float32 forced-count fallback only; no model/audio"},
              "random_uniform": rng.random_sample(24).tolist(), "cases": out}
    encoded = (json.dumps(result, separators=(",", ":")) + "\n").encode()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    print(f"wrote {len(out)} KMeans cases to {path}")


if __name__ == "__main__":
    main()
