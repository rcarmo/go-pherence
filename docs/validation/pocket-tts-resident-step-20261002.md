## Pocket TTS resident step gradient/metric reductions -- 2 October 2026

`ptx-resident-step-hybrid` and `vulkan-resident-step-hybrid` add utility tapes for step branch-gradient combination, logvar weighting, row metric reduction and final target input-gradient addition. CPU dispatch/boundary assembly, weighting-input assembly and final global norm-summary reduction remain, along with inter-tape transfers and completed-update full host snapshots. This is not a fused full-device step or a measured performance improvement.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. New source uncommitted; held inference index unchanged at eight paths/469 insertions. No native GPU/runtime initialisation/probe/launch/recovery, real-model, audio or listening run occurred. Execution/listening/publication holds remain.

## Reduction contract

The step collects immutable row-major boundaries. Diagonal condition/direct-target derivatives already contain pEqual/validCount from the diagonal prediction seed. Distillation condition/direct-target derivatives are unweighted; its parameter accumulation has already received (1-pEqual)/validCount exactly once. Both logvar boundaries are raw. The utility tape computes:

```
dCondition = diagonalCondition + distillCondition*((1-pEqual)/validCount)
dTarget = diagonalTarget + distillTarget*((1-pEqual)/validCount)
diagonalLogvar = rawDiagonalLogvar*(pEqual/validCount)
distillLogvar = rawDistillLogvar*((1-pEqual)/validCount)
```

Four row metrics are `[diagonalLoss, diagonalSquare, distillLoss, distillSquare]`. Mean reductions use existing row gather plus affine dot with ones, then F32 inverse valid count. FlowLoss combines weighted mean losses; totalLoss adds EOSLossWeight*EOS. Metrics remain named F64 fields but originate from F32 tape values; older scopes retain host F64 reduction arithmetic. Negative normalized losses are valid; raw-square/EOS metrics must be nonnegative.

After weighting/backbone backward, a second utility tape adds the direct target gradient to the conditioner input gradient. No parameter gradients are downloaded/accumulated by these utility tapes. Minimal-LSD endpoint input-only stop-gradient remains unchanged. Step reduction is admitted only with explicit flow/branch/loss backends, preventing partial CPU-path double-scaling. Shared owner checks protect scope/model/component continuity.

Bounds:1..1024 frames,1..16384 hidden/latent widths, exact finite boundaries, contiguous active prefix and at least one valid frame. Every inactive row input/output is zero; nonzero padding rejects, rather than silently participating in reduction. Exact finite objective weights/EOS and tape byte ceilings are validated. Inputs/results must not alias. Arithmetic overflow rejects through finite tape outputs before optimiser mutation; CPU legacy contract remains unchanged.

New scopes `*-resident-step-hybrid-v1` preserve all older replay identities. Row tapes still execute/close separately; host collects/copies outputs into boundary arrays and assembles weighting inputs. Compared to branch mode, six per-row scaling downloads are replaced by five step-reduction tensors and one final input gradient in the tiny two-valid-row fixture, leaving148 boundary tensors. Count is not bytes/latency; graph/intermediate allocations and host transfers remain. Dense embedding VJP, forward zero-seed adjoints and full snapshots still need measured optimisation. Logical resident/tape ceilings are not total native/host memory bounds.

## Offline checks

An independent small exact arithmetic reference covers all combined gradients/logvar seeds and five metrics with a padded row. Independent F64 checks at1/17/257/1024 frames and objective weights0/.25/.75/1 measure maximum metric error `1.61854441238507e-07` and gradient error `5.937272007372485e-08`. Standalone metric bound3e-7+2e-6*abs(reference), gradient2e-7 calibrate F32 inverse/reduction rounding; pinned fixture gates are unchanged. Native arithmetic remains unqualified.

Pinned PyTorch complete-step metrics/loss3e-5, all-gradient/input5e-5, parameter/EMA update4e-6 pass. Nine EOS/flow objective combinations and a shorter valid prefix compare every gradient/update against CPU. Both named scopes pass checkpoint/resume parameters/M/V/EMA/raw replay with stable residency and cross-scope restore rejection. Every complete-step tape construction failure position is injected: host/replay unchanged, no retry/checkpoint, retained owner. Admission covers inactive/wrong/removed backend, malformed shapes/config/padding, nonfinite/negative metrics, byte ceilings, input aliasing and final input-gradient failures.

A self-contained mathematical delegate found no structural double-scaling under the explicit raw/scaled boundary contract; it flagged hidden prior weighting and missing EOS/backbone paths as implementation risks. Full-step all-gradient/input tests cover those paths. No source/native approval implied.

Final gate: **00:54:48.885347758--00:55:26.282442625 UTC**, exit0. Twelve tests repeated ten times: **120 top-level+110 subtest passes**; race **12+11**; affected regressions **37+45**. No failures/skips.

Local Go1.26.3, GOENV off/empty GOFLAGS, GOPROXY/GOSUMDB off, GOMAXPROCS2, both GPU-disable flags, nice10, serial packages/tests,30-second inner/120-second outer deadlines. Model/CLI vet, host build, ARM64/RISC-V CLI cross-builds, layout and whitespace gates pass; foreign binaries not executed. Existing row/vector/affine/tape offline mocks regress. No kernel source/module change or offline recompilation needed. Initial missing brace and fixture metric identifier corrections did not change tolerances.

Scoped input/result/scope/owner coverage100%; builder96.4%; reduction evaluator93.3%; final input evaluator92.9%; selected package37.6%. Native execution coverage absent; no released-size fit/speedup/quality claim.

Evidence: `/workspace/exports/pocket-gpu-training-step-20261002/`. Remaining: all-device global norm and fused boundary/weighting-input orchestration; reduce dense embedding VJP/graph/activation churn, CPU transfer boundaries/full snapshots. Native trig/loss/released-size numerics/resources/performance and sampler/new-voice/listening qualification await coordination.
