## Pocket TTS resident row branch arithmetic -- 2 October 2026

`ptx-resident-branch-hybrid` and `vulkan-resident-branch-hybrid` add row interpolation and minimal-LSD adjoint branch utility tapes using existing disjoint vector/affine kernels. All trainable gradients remain device-accumulated. CPU step branch accumulation, scalar weighting/metric reduction and final global norm remain, along with inter-tape transfers and completed-update full host snapshots. This is not full-device orchestration or a measured optimisation.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Source uncommitted; held inference index unchanged at eight paths/469 insertions. No native GPU/runtime initialisation/probe/launch/recovery, real-model, audio or listening run occurred. Execution/listening/publication holds remain.

## Utility graphs

Seven fixed-arity kinds are exposed, not arbitrary caller graphs:

- `interpolate`: `xs=s*target+(1-s)*noise`, `desired=target-noise`.
- `advance`: `xt=xs+delta*v`, `dxdt=v+delta*timeJVP`.
- `negate`: endpoint residual seed `-dr`.
- `mixed`: `gv=dr+delta*endpointDInput`, `gt=delta*dr`, `ddelta=dot(dr,timeJVP)+dot(endpointDInput,v)`.
- `finish`: `dx=mixedDInput+endpointDInput`; target/noise=`s*dx`/`(1-s)*dx`; DS=`dot(dx,target-noise)+mixedDTimes[0]-ddelta`; DT=`mixedDTimes[1]+ddelta`.
- `diagonal-target`: `time*inputVJP-scaledPredictionSeed`.
- `scale`: scale one owned vector.

`training_step.go` uses tapes for diagonal interpolation, prediction seed scaling and direct-target derivative. Minimal-LSD uses interpolation/advancement/negation/mixed/finish tapes, preserving endpoint input-only VJP/no endpoint parameter accumulation. Distillation returned condition/target/logvar scaling also uses utility tapes; device parameter accumulation still receives its objective scale exactly once. Diagonal logvar scaling, step dZ/direct-target accumulation, residual reporting and metric reduction remain CPU.

Inputs have exact arity/finite lengths; channel count1..16384. Finish time input is2 scalars and delta derivative1; interpolation/delta/time parameters require[0,1], arbitrary scale must be finite. Tape byte ceiling is admitted before materialisation. Outputs are owned, finite and disjoint from inputs/each other; malformed results reject without fallback. RunUnboundTape provides the existing active borrow/failure retention. New scopes are `*-resident-branch-hybrid-v1`; all earlier scope arithmetic/replay remains unchanged. No native kernel source/module changed in this slice.

Dot products reuse one-row affine kernels and add separately reduced contributions. This changes diagnostic associativity versus the old CPU paired per-element delta sum. New replay identity makes that change explicit. DS also has a different addition grouping. These are floating-point reduction differences, not a symbolic-contract change. Pinned minimal-LSD DS/DT still pass2e-5; parameter/input fixtures retain their previous bounds.

Tiny two-valid-row complete step:148 boundary tensors downloaded (112 prior+36 row utility/scaling), with no parameter-gradient download/upload at update. This is tensor count, not bytes/latency. Per-row extra graphs/copies/allocations and dispatch overhead may make this slower; no performance claim. Dense embedding VJP, forward zero-seed adjoints, full snapshots and boundary transfers remain costs for later measurement.

## Evidence

Seven independent small arithmetic references cover all utility kinds. A separate independent F64 comparison at channels1/32/257/16384 measures max elementwise error `2.3283064365386963e-09` and combined dot error `3.8928263591309076e-05`. Standalone bounds1e-8 for vectors and2e-6+2e-5*abs(reference) for long diagnostic reductions reflect F32 accumulation. They do not widen pinned fixture bounds. Native arithmetic remains untested.

Pinned independent PyTorch minimal-LSD loss/raw square2e-6, values5e-6, condition/noise/target/time2e-5 and parameters3e-5 pass; full-step loss3e-5, gradients/input5e-5, update/EMA4e-6 unchanged. Nine complete-step objective variants cover EOS weights0/.1/1 and diagonal weights0/.75/1. Gradient count confirms no forward/endpoint parameter leakage. Both named scopes pass checkpoint/resume all parameters/M/V/EMA/replay with stable residency and cross-scope rejection. Every complete-step tape-construction position is failed in turn; host/replay unchanged, retry/checkpoint unavailable, owner retained. Inactive/wrong/removed branch owner, legacy injection, budget, arity, range and borrowed/overlapping results reject.

A self-contained formula delegate found no concrete algebraic bug, noting desired-velocity gradients must reach target/noise if used and diagnostic F32 reassociation is separate from symbolic correctness. Diagonal target uses `time*dXTime-dPrediction`; minimal endpoint flow has no desired target-noise loss, so its finish derivative is the interpolation path. No source/hardware approval was provided.

Final gate: **00:45:36.440609990--00:46:15.420105852 UTC**, exit0. Eleven tests repeated ten times: **110 top-level+180 subtest passes**. Race **11+18**; affected regressions **36+38**. No failures/skips. Initial hand-calculated expected-value errors and incomplete notice text were corrected without changing arithmetic/tolerances.

Local Go1.26.3, GOENV off/empty GOFLAGS, GOPROXY/GOSUMDB off, GOMAXPROCS2, both GPU-disable flags, nice10, serial packages/tests,30-second inner/120-second outer deadlines. Model/CLI vet, host build, ARM64/RISC-V CLI cross-builds, layout and whitespace gates pass; foreign binaries not executed. Regression includes offline vector/affine/tape mocks. No kernel compilation needed because kernels unchanged.

Scoped builder/helper/boundary/scope/owner coverage100%; resident evaluator92.3%; dispatch/boundary helper83.3%; selected package total37.6%. Native execution coverage absent. No released-size fit/speedup/listening/trained-voice acceptance.

Evidence: `/workspace/exports/pocket-gpu-training-branch-20261002/`. Remaining: step accumulation/scalar/metric device orchestration and all-device global norm; reduce graph/activation churn, dense embedding VJP and inter-tape transfers/full snapshots. Native trig/loss/released-size numerics/resources/performance and sampler/new-voice/listening qualification await coordination.
