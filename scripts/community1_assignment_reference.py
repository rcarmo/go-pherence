#!/usr/bin/env python3
"""Synthetic VBx centroids and cosine/rectangular assignment oracle.

Runs extracted pinned pyannote statements and SciPy assignment, CPU-only.
No trained checkpoints, media, clustering pipeline imports or GPU. SciPy solver
source checksum and installed revision are checked before generating fixtures.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

CLUSTER_SHA = "6031fb7c21277a7e9901ef2cdaed7d5cd69f7ef45508dc4b45e82ce0da3c8fba"
SOLVER_SHA = "73d78155990732311a116bc04f7f64d874c0f3793172e9aab12a476cb3d15c2c"
SCIPY_REV = "e4e854eaa8f18d807cd3496028e257e36caa93cc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clustering", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--vbx-fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path, sha in [(args.clustering, CLUSTER_SHA), (args.solver, SOLVER_SHA)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise SystemExit("source checksum mismatch")
    import numpy as np
    import scipy
    import scipy.version
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from einops import rearrange
    if scipy.version.git_revision != SCIPY_REV:
        raise SystemExit("SciPy revision mismatch")
    scope = dict(np=np, linear_sum_assignment=linear_sum_assignment, cdist=cdist, rearrange=rearrange)
    tree = ast.parse(Path(args.clustering).read_text())
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseClustering")
    constrained = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "constrained_argmax")
    exec(compile(ast.Module(body=[constrained], type_ignores=[]), args.clustering, "exec"), scope)
    vbx = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VBxClustering")
    call = next(n for n in vbx.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")

    def assignment_target(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)

    start = next(i for i, n in enumerate(call.body) if assignment_target(n, "W"))
    centroid = ast.parse("def reference_centroids(q, sp, train_embeddings, dimension):\n    pass").body[0]
    centroid.body = call.body[start:start+2] + ast.parse("return centroids").body
    ast.fix_missing_locations(centroid)
    exec(compile(ast.Module(body=[centroid], type_ignores=[]), args.clustering, "exec"), scope)
    start = next(i for i, n in enumerate(call.body) if assignment_target(n, "e2k_distance"))
    assign = ast.parse("def reference_assignment(self, embeddings, centroids, segmentations, constrained_assignment):\n    num_chunks, num_speakers, dimension = embeddings.shape").body[0]
    assign.body += call.body[start:]
    ast.fix_missing_locations(assign)
    exec(compile(ast.Module(body=[assign], type_ignores=[]), args.clustering, "exec"), scope)
    flat = lambda a: [None if np.isnan(x) else float(x) for x in np.asarray(a).flatten()]
    owner = SimpleNamespace(metric="cosine", constrained_argmax=lambda scores: scope["constrained_argmax"](None, scores))
    rng = np.random.default_rng(91381)
    centroids, matching, assignments = [], [], []
    for index in range(12):
        rows, dim, speakers = 3+index, 2+index%5, 1+index%6
        embeddings = rng.normal(size=(rows, dim))
        q = rng.uniform(.1, 1., size=(rows, speakers))
        q /= q.sum(1, keepdims=True)
        priors = q.mean(0)
        if speakers >= 3:
            priors[0] = 1e-7
            priors[1] = np.nextafter(1e-7, 1.) if index%2 else np.nextafter(1e-7, 0.)
            priors[2:] *= (1-priors[:2].sum())/priors[2:].sum()
        result = scope["reference_centroids"](q, priors, embeddings, dim)
        centroids.append(dict(name=f"centroid_{index}", config=dict(Rows=rows, Dimension=dim, Speakers=speakers), embeddings=flat(embeddings), q=flat(q), priors=flat(priors),
                              expected=flat(result), speaker_indices=np.where(priors>1e-7)[0].tolist(), weight_sums=flat(q[:,priors>1e-7].sum(0))))
    vbxfixture = json.loads(Path(args.vbx_fixture).read_text())
    p, v = vbxfixture["plda"][1], vbxfixture["vbx"][-1]
    assert v["name"] == "prepared_bridge"
    rows, dim = p["rows"], p["config"]["InputDim"]
    raw = np.array(p["input"]).reshape(rows, dim)
    q = np.array(v["iterations"][-1]["gamma"]).reshape(rows, v["speakers"])
    priors = np.array(v["iterations"][-1]["priors"])
    centers = scope["reference_centroids"](q, priors, raw, dim)
    centroids.append(dict(name="plda_vbx_bridge", config=dict(Rows=rows, Dimension=dim, Speakers=v["speakers"]), embeddings=flat(raw), q=flat(q), priors=flat(priors), expected=flat(centers), speaker_indices=np.where(priors>1e-7)[0].tolist(), weight_sums=flat(q[:,priors>1e-7].sum(0))))
    for rows, cols in [(1,1),(1,5),(2,1),(2,2),(3,2),(2,3),(3,3),(3,5),(5,3),(8,8),(8,64),(8,1)]:
        for style in ["constant", "integer_ties", "negative", "random", "partial_nan"]:
            x = rng.integers(-3,4,size=(2,rows,cols)).astype(np.float64)
            if style == "constant": x[:] = 1.
            if style == "negative": x = -np.abs(x)
            if style == "random": x = rng.normal(size=x.shape)
            if style == "partial_nan":
                x[0,0,0] = np.nan
                x[1] = np.nan  # filled from finite global minimum in first chunk
                if rows*cols == 1: x[0,0,0] = -1.
            hard = scope["constrained_argmax"](None,x)
            matching.append(dict(name=f"{rows}x{cols}_{style}", config=dict(Chunks=2, Speakers=rows, Clusters=cols), scores=flat(x), labels=hard.flatten().tolist()))
    for index in range(10):
        chunks, speakers, clusters, dim, frames = 2, 1+index%4, 1+index%5, 3+index%3, 5
        embs, cents = rng.normal(size=(chunks,speakers,dim)), rng.normal(size=(clusters,dim))
        if index == 8: cents[:] = cents[0]  # exact ties
        seg = rng.integers(0,2,size=(chunks,frames,speakers)).astype(np.float32)
        seg[0,:,0] = 0
        if index%2: seg[1,0,0] = np.nan
        for constrained in [False, True]:
            hard, soft, _ = scope["reference_assignment"](owner,embs,cents,SimpleNamespace(data=seg),constrained)
            assignments.append(dict(name=f"cosine_{index}_{constrained}", config=dict(Chunks=chunks,Speakers=speakers,Clusters=clusters,Dimension=dim,Frames=frames,Constrained=constrained), embeddings=flat(embs), centroids=flat(cents), segmentations=flat(seg), scores=flat(soft), labels=hard.flatten().tolist()))
    # Final bridge: PLDA -> VBx -> centroids using ORIGINAL embeddings -> scores.
    rows, dim = raw.shape
    seg = np.ones((1,5,rows),dtype=np.float32);seg[:,:,0]=0
    hard,soft,_ = scope["reference_assignment"](owner,raw[None],centers,SimpleNamespace(data=seg),True)
    assignments.append(dict(name="plda_vbx_centroid_assignment_bridge",config=dict(Chunks=1,Speakers=rows,Clusters=len(centers),Dimension=dim,Frames=5,Constrained=True),embeddings=flat(raw),centroids=flat(centers),segmentations=flat(seg),scores=flat(soft),labels=hard.flatten().tolist()))
    result = dict(schema=1, absolute_tolerance=2e-10, relative_tolerance=2e-10, reference=dict(clustering_sha256=CLUSTER_SHA,solver_sha256=SOLVER_SHA,scipy_revision=SCIPY_REV,numpy=np.__version__,scipy=scipy.__version__), scope="synthetic float64 centroids and assignment; no AHC/full pipeline/trained quality", centroids=centroids, matching=matching, assignments=assignments)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"wrote {len(centroids)} centroid/{len(matching)} matching/{len(assignments)} cosine cases")


if __name__ == "__main__":
    main()
