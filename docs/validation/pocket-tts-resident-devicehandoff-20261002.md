## Opt-in retained-device training handoff

`ptx-resident-devicehandoff-hybrid-v1` and `vulkan-resident-devicehandoff-hybrid-v1` integrate the private retained-device diagonal and distillation helpers into the training step. Forward hidden/vector downloads and original host overflow guards remain. These experimental scopes require explicit device opt-in, a runtime SHA256 pin and conservative byte ceilings; native execution has not run.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths / 469 insertions. GPU/model/audio/listening/publication holds are unchanged.

## Coordinator ownership and routing

The scopes inherit EOS admission, complete source/binding admission, compact embedding gradients, pinned source leases, lifetime/zero storage and device optimiser coordination. Existing scope predicates and numerical policies retain their old routes. Public calls without an active step lease use the existing public graph/download path.

`beginDeviceCondition` accepts only the coordinator's exact active hidden-row slice. It creates a read-only row-condition child from the pinned transformer/EOS root. Private diagonal/distillation constructors then capture model/root/row/time provenance and frozen arithmetic metadata and pin their forward child outputs. No caller-supplied lease can replace the coordinator's root or row.

Diagonal forward still downloads input/desired/prediction; distillation forward still downloads all six vectors. Output ownership/shape/finite checks and the original F64 LSD overflow guard run before the values reach step orchestration. The coordinator records slice identity and immutable copies for backward provenance. Equality here checks modification of the same admitted host arrays, not generated-signal equivalence against a different implementation.

Backward validates row/kind, exact host array identities and unchanged values; distillation also checks S/T and target/noise against captured provenance. Two finite/disjoint scalar cutoff/witness outputs precede each private backward accumulation. Shape-only sources bind the actual retained device roots, while host arrays continue to supply the redundant guard and public boundary contract.

Diagonal keeps `time*dX - scaled prediction seed`, raw statistics and unit coordinator accumulation for the already-scaled seed. Distillation exposes only primary parameter gradient roots, applies objective scaling once at accumulation and retains endpoint stop-gradient. Raw input derivatives remain unscaled. Reduction, weighting, EOS-backbone and completed-update synchronisation retain their established contracts.

Successful backward closes the exact forward child before returning a row, then closes the row-condition child. The transformer parent closes before reduction/update. The close guard rejects any live prediction or retained-device child. All supported calls are serialised by the caller.

## Failure and publication boundaries

Backend failure preserves the first terminal error and calls the root lease's `RetainFailure`. Partial construction, cancellation, rejected scalar flags, malformed/borrowed results and child/condition/parent close errors retain uncertain resources. Previously accumulated gradients may remain; no rollback is asserted.

Training-step cleanup clears successful metrics/gradient publication on failure. Runner tests verify unchanged trainer parameters/moments/EMA and replay state, rejection of checkpoint/device snapshot/update/retry, and no further tape construction on retry. Failures at every tape construction and a late second-row rejection exercise the publication boundary. Active public operations and cross-scope restore are rejected.

## CLI and replay identity

Unversioned CLI names are `ptx-resident-devicehandoff-hybrid` and `vulkan-resident-devicehandoff-hybrid`. Their replay kind is `resident-devicehandoff-backend`; runtime pin and scope participate in the existing options fingerprint. Factory case coverage verifies both names without native initialisation, and cancellation rejects before backend opening. The notice states that host downloads/guards, step results and full snapshots remain and native qualification is pending.

Same-scope checkpoint/resume tests compare parameters, M/V, EMA and replay state after multiple updates. Cross-scope restore from the old pruned scope fails without mutation. Legacy identities and builder ownership remain unchanged.

## Numerical and accounting evidence

Full-step tests compare pinned PyTorch metrics, every gradient, AdamW updates and EMA for PTX/Vulkan mocks. Further tests cover objective weights 0/0.25/1, EOS weights 0/0.1/1, padding, repeated and empty token sequences, clipping ceilings 0.01/0.1/unclipped and resume.

Independent retained-source references keep their existing tolerances:

| Reference | Comparisons | Mean absolute error | RMS error | Maximum absolute error |
|---|---:|---:|---:|---:|
| Diagonal CPU VJP / F64 loss | 10,530 | 8.27725061e-10 | 6.15896965e-9 | 2.38418579e-7 |
| Distillation CPU mixed AD / F64 loss | 4,728 | 1.86024773e-10 | 2.07511406e-9 | 8.97474233e-8 |

Gradient gates remain 5e-5; loss statistics use `2e-6 + 2e-6*abs(reference)`. Full-step pinned gates are unchanged. Generated-output acceptance uses numerical comparisons, without hash equality gates.

| Tiny complete mock step | Prediction-handoff v1 | Device-handoff v1 |
|---|---:|---:|
| Tapes | 11 | 19 |
| Commands | 3,707 | 3,773 |
| Downloads | 47 | 53 |
| Downloaded elements | 122 | 132 |
| Parameter accumulations | 161 | 161 |
| Graph logical bytes | 134,240 | 135,112 |
| Planned tape storage bytes | 42,268 | 42,556 |

The new row-condition children, separate scalar admission and explicit distillation noise/time decoding add overhead. Forward vector downloads are retained; full snapshots remain. These numbers measure offline mock accounting, not native allocator peak, transferred latency or complete-request performance. No speedup has been demonstrated.

## Final offline gates

Window: **2026-10-02 16:08:07--16:10:10 UTC**. GPUs and Go network access disabled; `GOMAXPROCS=2`, tests `-p=1 -parallel=1`.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 460 | 2,560 | 0 | 0 |
| Race detector | 46 | 256 | 0 | 0 |
| CPU/mock regression | 1,019 | 1,886 | 0 | 1 |
| Compiled-artifact follow-up | 46 | 256 | 0 | 0 |

The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` skipped. Vet, host build, Linux ARM64/RISC-V builds, model layout, diff checks and offline PTX/SPIR-V compilation/validation passed. Kernels and ABI are unchanged. Race tests exercise serialised ownership, not concurrent lease use.

Scoped coverage: scope/condition/provenance/lifetime helpers and diagonal backward 100%; diagonal forward 95%, distillation forward 100%, distillation backward 93.3%. Some owner-download/internal close-error branches and native execution remain unqualified. Initial test fixture API errors, missing CLI admission/factory cases and accounting expectations were corrected before final passing gates. The first passing broad window is preserved under `initial-pass/`.

A focused read-only source judge timed out after 120 seconds without a verdict. A subsequent bounded, tool-free review of the final coordinator slice and explicitly supplied helper contracts found no concrete blocker. It noted that equal-content target/noise replacement arrays pass the provenance check; backward uses the captured owned copies, so those replacement arrays cannot alter device arithmetic. This review did not inspect the full source or qualify native execution. `@default` subsequently read the five coordinator/private-helper files completely and traced the related lease/runner/admission/publication paths. It found no additional concrete lifetime, admission-order or scaling defect with host guards retained and independently counted the saved regression/race events. The scoped review confirmed the released-size resident admission blocker; it grants no full-source/native approval. Stale helper scope comments were corrected without runtime changes. Review note: `/workspace/exports/pocket-devicehandoff-independent-review-20261002.md`. All execution holds remain. Evidence: `/workspace/exports/pocket-gpu-training-devicehandoff-20261002`; script: `/workspace/tmp/pocket-devicehandoff-gate.sh`.

## Remaining work

[Released resource audit](pocket-tts-resident-releasedresources-20261002.md) confirms 2,866,551,644 resident logical bytes exceed the current 2 GiB CLI maximum, and the retained distillation pair alone exceeds the default 256 MiB live-source ceiling. Runtime limits are unchanged. Forward download removal needs further source/native qualification of the scalar admission contracts and a separate explicit download policy; current host guards cannot be silently bypassed. Existing hidden/backward result transfers, coherent snapshots and dense persistent optimiser work remain. Native resources, transfer-inclusive performance, released-size voice training and listening acceptance have not been verified. The overall implementation goal is incomplete.
