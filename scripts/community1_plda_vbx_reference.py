#!/usr/bin/env python3
"""Synthetic float64 prepared-PLDA and deterministic VBx oracles.

Extract pinned source functions. Single CPU thread, no trained assets/media/GPU.
Actual vbx_setup reads temporary synthetic npz, performs SciPy eigensetup, then
we export its prepared coefficients. The Go runtime does not use Python/SciPy.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile

VBX_SHA = "a8c644feea4b381f9c1e7da72e0e47775c1fd482067e686801ddc16e5cac3c0e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbx", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if hashlib.sha256(Path(args.vbx).read_bytes()).hexdigest() != VBX_SHA:
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
    flat = lambda x: np.asarray(x).flatten().tolist()
    rng = np.random.default_rng(9217)
    plda_cases, vbx_cases = [], []
    with tempfile.TemporaryDirectory(prefix="synthetic-plda-") as tmp:
        for inp, proj, out, rows in [(3, 2, 1, 4), (5, 3, 3, 7), (8, 4, 2, 9), (2, 2, 2, 3)]:
            mean1, mean2, mu = rng.normal(size=inp) * .1, rng.normal(size=proj) * .1, rng.normal(size=proj) * .1
            lda = rng.normal(size=(inp, proj)) * .4
            tr = np.diag(np.linspace(.8, 1.4, proj)) + rng.normal(size=(proj, proj)) * .03
            psi = np.linspace(.2, 1.7, proj)
            embeddings = rng.normal(size=(rows, inp))
            transform_path, plda_path = Path(tmp) / "xvec.npz", Path(tmp) / "plda.npz"
            np.savez(transform_path, mean1=mean1, mean2=mean2, lda=lda)
            np.savez(plda_path, mu=mu, tr=tr, psi=psi)
            xvec_tf, plda_tf, phi = scope["vbx_setup"](transform_path, plda_path)
            # Obtain exact prepared transform captured by the real source lambda.
            captured = dict(zip(plda_tf.__code__.co_freevars, [cell.cell_contents for cell in plda_tf.__closure__]))
            prepared = captured["plda_tr"]
            intermediate = xvec_tf(embeddings)
            result = plda_tf(intermediate, lda_dim=out)
            plda_cases.append(dict(config=dict(InputDim=inp, ProjectedDim=proj, OutputDim=out), rows=rows,
                                   weights=dict(Mean1=flat(mean1), Mean2=flat(mean2), LDA=flat(lda), Mu=flat(mu), Transform=flat(prepared), Phi=flat(phi)),
                                   input=flat(embeddings), output=flat(result), phi=flat(phi[:out])))

    for name, rows, dim, labels, fa, fb, smooth, iterations, epsilon in [
        ("two_groups", 8, 3, [0]*4+[1]*4, .07, .8, 7., 20, 1e-4),
        ("single_speaker", 5, 2, [0]*5, .07, .8, 7., 20, 1e-4),
        ("gapped_labels", 7, 4, [0, 2, 2, 0, 2, 0, 2], .2, .7, 7., 8, 1e-4),
        ("hard_init", 9, 5, [0, 1, 2]*3, .15, 1., -1., 12, 1e-4),
        ("uniform_init", 6, 4, [0, 1]*3, .07, .8, 0., 20, 1e-4),
        ("one_iteration", 6, 3, [0, 1, 2]*2, .07, .8, 7., 1, 1e-4),
        ("zero_phi", 6, 3, [0, 1]*3, .07, .8, 7., 20, 1e-4),
        ("separated_large", 12, 5, [0]*6+[1]*6, .07, .8, 1000., 10, 0.),
        ("forced_stop", 8, 3, [0, 1]*4, .07, .8, 7., 20, 1e6),
        ("prepared_bridge", 7, 3, [0, 0, 0, 1, 1, 1, 1], .07, .8, 7., 20, 1e-4),
    ]:
        x = rng.normal(size=(rows, dim)) * .4
        for i, label in enumerate(labels):
            x[i] += label * (10 if name == "separated_large" else 1.2)
        phi = np.linspace(1.2, .1, dim) if name != "zero_phi" else np.zeros(dim)
        if name == "prepared_bridge":
            x = np.array(plda_cases[1]["output"]).reshape(rows, dim)
            phi = np.array(plda_cases[1]["phi"])
        speakers = max(labels)+1
        qinit = np.zeros((rows, speakers))
        qinit[np.arange(rows), labels] = 1.
        if smooth >= 0:
            qinit = softmax(qinit * smooth, axis=1)
        # Instrument only the append boundary; all update/stop arithmetic remains
        # the pinned function. These are actual iteration states, not a rewrite.
        fn = next(n for n in ast.parse(Path(args.vbx).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "VBx")
        class Capture(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "Li" and node.value.func.attr == "append":
                    extra = ast.parse("capture(gamma.copy(), pi.copy(), alpha.copy(), invL.copy(), ELBO)").body[0]
                    return [node, extra]
                return node
        captured = []
        scope["capture"] = lambda q, p, a, inv, elbo: captured.append(dict(gamma=flat(q), priors=flat(p), alpha=flat(a), inv_l=flat(inv), elbo=float(elbo)))
        instrumented = Capture().visit(fn)
        instrumented.name = "observed_vbx"
        ast.fix_missing_locations(instrumented)
        exec(compile(ast.Module(body=[instrumented], type_ignores=[]), args.vbx, "exec"), scope)
        gamma, priors, elbo, alpha, inv_l = scope["observed_vbx"](x, phi, Fa=fa, Fb=fb, pi=speakers, gamma=qinit, maxIters=iterations, epsilon=epsilon, return_model=True)
        raw = scope["VBx"](x, phi, Fa=fa, Fb=fb, pi=speakers, gamma=qinit, maxIters=iterations, epsilon=epsilon, return_model=True)
        assert all(np.array_equal(a,b) for a,b in zip(raw,(gamma,priors,elbo,alpha,inv_l)))
        if epsilon == 1e-4:
            wrapper_q, wrapper_pi = scope["cluster_vbx"](np.array(labels), x, phi, fa, fb, maxIters=iterations, init_smoothing=smooth)
            assert np.array_equal(gamma, wrapper_q) and np.array_equal(priors, wrapper_pi)
        vbx_cases.append(dict(name=name, config=dict(Rows=rows, Dimension=dim, MaxIterations=iterations, Fa=fa, Fb=fb, Epsilon=epsilon, InitSmoothing=smooth),
                              input=flat(x), phi=flat(phi), labels=labels, speakers=speakers, iterations=captured))
    result = dict(schema=1, absolute_tolerance=2e-10, relative_tolerance=2e-10,
                  reference=dict(vbx_sha256=VBX_SHA, pyannote_commit="b749285c5cdd4636b2edc7f766f1352c8dde9369", numpy=np.__version__, scipy=scipy.__version__, dtype="float64", cpu_threads=1),
                  scope="synthetic float64 prepared transform and deterministic GMM VBx; no native checkpoint setup/AHC/trained quality", plda=plda_cases, vbx=vbx_cases)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"wrote {len(plda_cases)} PLDA and {len(vbx_cases)} VBx cases/{sum(len(c['iterations']) for c in vbx_cases)} iteration states")


if __name__ == "__main__":
    main()
