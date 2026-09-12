#!/usr/bin/env python3
"""Pinned SciPy float64 centroid-linkage AHC oracle, synthetic CPU only.

No trained model/media/GPU. Checks installed SciPy revision and source hashes;
includes tie/inversion/threshold cases and an AHC->PLDA->VBx->assignment bridge.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REV = "e4e854eaa8f18d807cd3496028e257e36caa93cc"
HASHES = {"hierarchy": "50141ab68a04ee82d51bbba906afdf29a7a79877223f9e89699500e926dabf9f",
          "updates": "64d6c695524fc08fc1d4c9679bd4424b28c15232b9776df4224ce432b8ee725f",
          "structures": "1bf594add2b5f92d07a0649cf8fd54ae5b2352a75ce07dcc462d82a821493e3d",
          "vbx": "a8c644feea4b381f9c1e7da72e0e47775c1fd482067e686801ddc16e5cac3c0e"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in HASHES:
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--plda-fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for name, sha in HASHES.items():
        if hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{name} checksum mismatch")
    import numpy as np
    import scipy
    import scipy.version
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.special import logsumexp, softmax
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    if scipy.version.git_revision != REV:
        raise SystemExit("SciPy revision mismatch")
    flat = lambda a: np.asarray(a).flatten().tolist()
    cases = []

    def run(name, x, threshold):
        x = np.asarray(x, dtype=np.float64)
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        z = linkage(unit, method="centroid", metric="euclidean")
        labels = fcluster(z, threshold, criterion="distance") - 1
        _, labels = np.unique(labels, return_inverse=True)
        cases.append(dict(name=name, config=dict(Rows=len(x), Dimension=x.shape[1], Threshold=float(threshold)), input=flat(x),
                          linkage=[dict(Left=int(a), Right=int(b), Distance=float(d), Size=int(s)) for a,b,d,s in z], labels=labels.tolist(), clusters=int(labels.max()+1)))
        return labels

    for label, threshold in [("below", np.nextafter(2., 0.)), ("equal", 2.), ("above", np.nextafter(2., 3.))]:
        run("boundary_"+label, [[1.,0.],[-1.,0.]], threshold)
    for n in [2,3,4,8,17]:
        for threshold in [0., .6]:
            run(f"duplicates_{n}_{threshold}", np.tile([1.,2.,3.], (n,1)), threshold)
    for name,x in [("axis_ties",np.eye(4)),("opposite_ties",[[1,0],[0,1],[-1,0],[0,-1]]),
                   ("equilateral_inversion",[[1.,0.],[-.5,np.sqrt(3)/2],[-.5,-np.sqrt(3)/2]])]:
        for threshold in [.6,1.3,1.6,1.8]:
            run(f"{name}_{threshold}",x,threshold)
    rng = np.random.default_rng(92378)
    for i in range(16):
        n,d = 5+i*2, 2+i%7
        x = rng.normal(size=(n,d))
        if i%3==0:
            x[n//2:] += 3
        run(f"seeded_{i}",x,[.3,.6,1.,1.5][i%4])
    prepared = json.loads(Path(args.plda_fixture).read_text())["plda"][1]
    x = np.array(prepared["input"]).reshape(prepared["rows"],prepared["config"]["InputDim"])
    labels = run("pipeline_bridge",x,.6)
    scope = dict(np=np,logsumexp=logsumexp,softmax=softmax)
    tree = ast.parse(Path(args.vbx).read_text())
    for name in ["VBx","cluster_vbx"]:
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),args.vbx,"exec"),scope)
    features = np.array(prepared["output"]).reshape(len(x),prepared["config"]["OutputDim"])
    q,pi = scope["cluster_vbx"](labels,features,np.array(prepared["phi"]),Fa=.07,Fb=.8)
    w = q[:,pi>1e-7]
    centers = w.T@x/w.sum(0)[:,None]
    scores = 2-cdist(x,centers,metric="cosine")
    row,col = linear_sum_assignment(scores,maximize=True)
    hard = -2*np.ones(len(x),dtype=int);hard[row]=col
    bridge = dict(gamma=flat(q),priors=flat(pi),centroids=flat(centers),scores=flat(scores),labels=flat(hard),speakers=len(pi),clusters=len(centers))
    result = dict(schema=1,absolute_tolerance=2e-10,relative_tolerance=2e-10,reference=dict(scipy_revision=REV,source_hashes=HASHES,numpy=np.__version__,scipy=scipy.__version__,dtype="float64"),
                  scope="synthetic bounded centroid-linkage and prepared full clustering component bridge, not neural pipeline",cases=cases,bridge=bridge)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"wrote {len(cases)} AHC cases, {sum(len(c['linkage']) for c in cases)} merges, {sum(any(c['linkage'][i]['Distance']<c['linkage'][i-1]['Distance'] for i in range(1,len(c['linkage']))) for c in cases)} inversion cases")


if __name__ == "__main__":
    main()
