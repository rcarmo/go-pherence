## Private retained-device distillation backward

The private distillation path retains six forward roots and checks two scalar admission outputs before accumulating primary parameter gradients. Condition and forward values are shape-only read-only sources; no forward vectors download. Existing production scopes, redundant host guards and CLI/replay identities are unchanged. The new helpers have no production caller.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths / 469 insertions. GPU/model/audio/listening/publication holds are unchanged.

## Retained forward and source admission

`beginDistillDeviceSources` builds the existing host-admitted distillation forward tape with captured arithmetic metadata. Value-owned topology/epsilons and copied frequencies use `freezeDiagonalDeviceFlow`; target/noise values receive private owned copies for backward finish arithmetic. Both forward and backward rebind parameters from the same pinned resident state, which cannot update or restore while the lease is live.

The forward graph has two condition constants. After full source/binding admission, the helper declares them external, remaps both read paths to one retained condition source, and removes the unused external declaration. Its dead extent remains charged to the complete logical budget, with no invented host values. Condition discovery follows only reshape aliases and fails closed unless it finds exactly two known source constants.

The private provenance record identifies the model, parent root, row, S/T values, extents, condition root and six exact child roots: primary input, velocity, tangent, endpoint input, desired and prediction. Caller serialisation is required. A shared single-attempt marker prevents copied descriptors from accumulating a second time. All roots must pass the resident lease tree's exact identity/liveness/extent/alias and combined live-source checks.

`buildDistillDeviceAdmission` reads all seven roots without gradients. The existing LSD mode2 cutoff checks prediction versus desired; a finite witness checks condition and every forward root. Two distinct finite scalar zeros are required before constructing/executing backward. NaN/Inf in each actual source, excessive differences, invalid flags and cancellation reject before accumulation. No fabricated host vectors supply numerical approval.

## Backward composition and stop-gradient

`buildEndpointDeviceTape` binds condition, endpoint input, desired, prediction, velocity and tangent externally. It preserves the endpoint input-only VJP and mixed seed/delta arithmetic. `buildPrimaryDeviceTape` binds condition and primary input externally; its four seed descriptors are explicit shape-only sources until composition connects them to endpoint results.

The existing `composeDistillTrainingTapes` admits both complete graphs, shares parameter IDs and exposes only primary parameter gradient roots. The private wrapper then clears the unused external declarations of remapped primary seeds and merges the second condition read identity. Dead extents stay in full-source admission. Parameter and input-source inventories are revalidated before pruning.

Only primary parameter gradients accumulate; endpoint parameter gradients are never exposed or added. Coordinator `parameterScale` applies once. Input/condition/target/noise/time derivatives and raw statistics stay unscaled. The primary finish preserves `dx = primary input adjoint + endpoint input adjoint`, target/noise interpolation derivatives and the original S/T delta contributions.

The runner checks five selected results for shape, finiteness, distinct storage and nonnegative square. It closes the exact forward child before returning the row; the parent stays pinned across rows. Admission, result or close failures clear the result and terminally retain the tree. A late failure may leave earlier or current parameter gradients present, but cannot publish a successful row, snapshot, update or retry. Uncertain owners are deliberately retained rather than blindly freed.

## Offline checks

36 combinations cover S/T pairs (0/0, 0/1, 0.3/0.8, 1/1), logvar -2/0/1 and parameter scale 0/0.25/1. Results match the existing composed distillation graph; update preflight observes only scaled primary gradients. Metadata and caller target/noise edits after forward construction do not change captured backward arithmetic.

Independent CPU references compute interpolation, primary selected-time JVP, endpoint input-only reverse, primary mixed AD and finish derivatives. Loss statistics use the F64 normalized-square formula.

| Comparisons | Mean absolute error | RMS error | Maximum absolute error |
|---:|---:|---:|---:|
| 4,728 | 1.86024773e-10 | 2.07511406e-9 | 8.97474233e-8 |

Unchanged gates: gradient comparisons 5e-5; loss statistics `2e-6 + 2e-6*abs(reference)`. Tests use numerical comparisons, without generated-output hash gates.

The tiny composed backward graph has 889 commands, 1,099 tensors, seven external roots and 34 primary parameter bindings. Full-source logical storage is 24,312 bytes; exact admission passes and one byte less rejects. Each successful row makes two mock download calls: two admission elements and twelve result elements (condition3/target2/noise2/time2/statistics3). There are no forward-source or parameter-gradient downloads before update. Native peak memory, transfer-inclusive latency and full-request performance are unmeasured.

Tests cover malformed topology/extents/times/logvar/scale/budgets, source declarations, boundary count/shape, exact and foreign/closed owners, single-use copies, cancellation before begin and after admission, all source NaN/Inf cases, malformed/aliased flags or results, negative square and child-close failure. A late second-row admission failure preserves the first row's accumulation count and fences result, update, snapshot, retry and close.

## Final offline gates

Window: **2026-10-02 15:41:20--15:42:44 UTC**. GPUs and Go network access disabled; `GOMAXPROCS=2`, tests `-p=1 -parallel=1`.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 470 | 2,080 | 0 | 0 |
| Race detector | 47 | 208 | 0 | 0 |
| CPU/mock regression | 995 | 1,788 | 0 | 1 |
| Compiled-artifact follow-up | 47 | 208 | 0 | 0 |

The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` skipped. Vet, host build, Linux ARM64/RISC-V builds, model layout, diff checks and offline PTX/SPIR-V compilation/validation passed. Kernels and ABI are unchanged. Race tests exercise supported serialised ownership.

Scoped coverage: metadata/source-ID validation, edge mapping and scalar admission 100%; endpoint builder 92.6%, primary builder 91.4%, composition 89.7%, owner constructor 91.2%, forward-root validation 92.3%, runner 91.3%. Defensive internal-builder/discovery branches and native execution are not fully exercised. A focused read-only source judge timed out after 120 seconds without a verdict; independent source/native approval was not obtained.

Evidence: `/workspace/exports/pocket-gpu-training-distilldeviceinputs-20261002`; script: `/workspace/tmp/pocket-distilldeviceinputs-gate.sh`. The first passing window is preserved under `initial-pass/`.

## Next integration

A separate production opt-in route still needs coordinator ownership, exact row-condition provenance, step reduction, full-step/replay/checkpoint failure tests and explicit qualification before any new download policy. Existing scopes retain host hidden/prediction/desired/input/backward downloads and redundant guards. Coherent snapshots, dense persistent optimiser work, native qualification and voice/listening acceptance are unfinished.
