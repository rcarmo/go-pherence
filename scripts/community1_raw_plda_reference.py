#!/usr/bin/env python3
"""Synthetic raw PLDA preparation and numeric NPZ interoperability oracle.

Single CPU thread, no trained models/media/GPU. Executes pinned vbx_setup and
cluster_vbx; compares coordinates only up to eigenspace basis invariance.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile

SHA = "a8c644feea4b381f9c1e7da72e0e47775c1fd482067e686801ddc16e5cac3c0e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--npz-dir", required=True)
    args = parser.parse_args()
    if hashlib.sha256(Path(args.vbx).read_bytes()).hexdigest() != SHA:
        raise SystemExit("source checksum mismatch")
    import numpy as np
    import scipy
    from scipy.linalg import eigh
    from scipy.special import logsumexp, softmax
    scope = dict(np=np, eigh=eigh, logsumexp=logsumexp, softmax=softmax)
    tree = ast.parse(Path(args.vbx).read_text())
    for name in ["l2_norm", "vbx_setup", "VBx", "cluster_vbx"]:
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), args.vbx, "exec"), scope)
    flat = lambda a: np.asarray(a).flatten().tolist()
    rng = np.random.default_rng(9304)
    cases = []
    with tempfile.TemporaryDirectory(prefix="synthetic-raw-plda-") as tmp:
        for index, (inp, dim, out) in enumerate([(4,3,2),(5,3,3),(6,4,2),(3,3,3),(8,5,4),(2,1,1)]):
            mean1,mean2,mu = rng.normal(size=inp)*.1,rng.normal(size=dim)*.1,rng.normal(size=dim)*.1
            lda = rng.normal(size=(inp,dim))*.4
            tr = rng.normal(size=(dim,dim))*.1+np.diag(np.linspace(.7,1.3,dim))
            psi = rng.permutation(np.linspace(.2,1.5,dim))
            if index == 1: psi[:] = .7  # full exact degenerate block
            if index == 2: psi[:] = [2, .3, 2, .3]  # whole leading repeated block
            if index == 3: tr = tr[::-1]  # pivot row swaps
            embeddings = rng.normal(size=(7,inp))
            xp,pp = Path(tmp)/"xvec.npz",Path(tmp)/"plda.npz"
            np.savez(xp,mean1=mean1,mean2=mean2,lda=lda)
            np.savez(pp,mu=mu,tr=tr,psi=psi)
            xt,pt,phi = scope["vbx_setup"](xp,pp)
            reference = pt(xt(embeddings),lda_dim=out)
            order = np.argsort(-psi,kind="stable")
            direct = (xt(embeddings)-mu)@tr[order].T
            direct = direct[:,:out]
            # Finite-precision eigenvectors differ, but retained feature Gram
            # matrices and VBx responsibilities must remain equivalent.
            np.testing.assert_allclose(direct@direct.T,reference@reference.T,atol=2e-10,rtol=2e-10)
            labels = np.array([0,0,0,1,1,1,1])
            q,pi = scope["cluster_vbx"](labels,reference,phi[:out],Fa=.07,Fb=.8)
            q2,pi2 = scope["cluster_vbx"](labels,direct,psi[order][:out],Fa=.07,Fb=.8)
            np.testing.assert_allclose(q,q2,atol=2e-10,rtol=2e-10)
            np.testing.assert_allclose(pi,pi2,atol=2e-10,rtol=2e-10)
            cases.append(dict(config=dict(InputDim=inp,ProjectedDim=dim,OutputDim=out),rows=7,raw=dict(Mean1=flat(mean1),Mean2=flat(mean2),LDA=flat(lda),Mu=flat(mu),TR=flat(tr),Psi=flat(psi)),input=flat(embeddings),order=flat(order),direct=flat(direct),gram=flat(reference@reference.T),phi=flat(phi[:out]),gamma=flat(q),priors=flat(pi),condition=float(np.linalg.cond(tr,np.inf))))
            if index == 0:
                directory = Path(args.npz_dir);directory.mkdir(parents=True,exist_ok=True)
                for dtype in ["<f8",">f8","<f4",">f4"]:
                    tag = ("le" if dtype[0]=="<" else "be")+dtype[1:]
                    convert = lambda a: np.asarray(a,dtype=dtype)
                    np.savez_compressed(directory/(tag+"-xvec.npz"),mean1=convert(mean1),mean2=convert(mean2),lda=convert(lda))
                    np.savez(directory/(tag+"-plda.npz"),mu=convert(mu),tr=convert(tr),psi=convert(psi))
    result = dict(schema=1,absolute_tolerance=2e-10,relative_tolerance=2e-10,reference=dict(vbx_sha256=SHA,numpy=np.__version__,scipy=scipy.__version__,dtype="float64"),scope="synthetic positive full-rank raw preparation, basis-invariant VBx; not actual checkpoint or mixed-dtype quality",cases=cases)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"wrote {len(cases)} raw PLDA cases and 8 synthetic NumPy NPZ archives")


if __name__ == "__main__":
    main()
