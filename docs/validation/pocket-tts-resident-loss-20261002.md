## Pocket TTS resident LSD losses -- 2 October 2026

`ptx-resident-loss-hybrid` and `vulkan-resident-loss-hybrid` evaluate normalized diagonal and minimal-LSD row losses, raw squares and prediction/logvar seeds inside device tapes. They inherit EOS/time/sequence/projection/flow/transformer/weighting and AdamW/EMA ownership. All trainable parameter gradients stay resident. CPU interpolation/branch orchestration, valid-row/objective scaling, final global norm reduction, inter-tape transfers and completed-update full host snapshots remain. Full-device training is unfinished.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. New source is uncommitted; held inference index remains eight paths/469 insertions. No native GPU/runtime initialisation/probe/launch/recovery, real-model, audio or listening run occurred. Execution/listening/publication holds remain.

## Row loss and minimal endpoint contract

`lsd-loss` consumes prediction[C], desired[C] and immutable logvar[1]. It writes statistics `[rawSquare, loss, dLogvar]` or prediction seed[C]. One thread computes compensated F32 square sums; each seed element owns its output. There are no atomics or in-place output writes.

```
difference = prediction - desired
square = sum(difference*difference)
scale = exp(logvar)/C
normalized = square*scale
loss = normalized-logvar
dLogvar = normalized-1
dPrediction = (2*difference)*scale
```

These signs preserve the pinned released implementation. Despite the log-variance name, the weighting behaves like log-precision; it must not be changed to conventional Gaussian-NLL signs without changing the upstream contract. CUDA uses expf, GLSL uses Exp; surrounding arithmetic disables contraction/uses precise values. Each loss is one row; host orchestration applies valid-row reduction and objective weights once. Older scopes retain CPU F64 diagonal loss and F32 minimal-LSD square arithmetic.

In minimal LSD, prediction is `v+(t-s)*timeJVP`, desired is endpoint output. Endpoint VJP is seeded with negative residual derivative, contributes only its input Jacobian, and never accumulates endpoint parameter/condition/time gradients. Primary value/tangent seeds remain `dr+delta*endpointDInput` and `delta*dr`. Device parameter accumulation gets objective scale once; returned condition/target/logvar gradients get the same scale once in the caller. Interpolation, branch sums and diagnostic time derivatives still run CPU. The unchanged all-parameter gradient count checks forward/endpoint non-accumulation.

Bounds:1..16384 channels, finite exact inputs and logvar within[-80,80]. Host F64 admission checks a conservative per-difference ceiling leaving4x raw/exp-weighted reduction headroom and seed headroom. This is overflow admission, not the objective computation. Byte ceiling is admitted before copies. Raw kernel users must supply these bounded inputs themselves; tapes verify them. Nonfinite statistics/seeds reject before any optimiser update. Negative loss is allowed, unlike BCE.

New scopes `*-resident-loss-hybrid-v1` bind the shared loss owner. Legacy injection/removal and cross-scope restore reject before host/replay mutation. Loss utility tapes use RunUnboundTape borrow, terminal retention, output validation and no parameter bindings/accumulation. The two-valid-row fixture downloads112 boundary tensors (104 prior +8 row-loss statistics/seeds), with no parameter-gradient download/upload at update. Count is not bytes or latency; loss predictions, scalars and seeds still transfer CPU↔device. Logical resident/tape ceilings do not bound total native/host memory.

## Numerical checks

Independent F64 calculus:25 cases (channels1/3/32/257/16384 × logvar-8/-1/0/1/8). F32 angle-independent differences are squared in F64 reference; maximum errors: square `9.086393220059108e-07`, normalized loss `7.711420010991787e-06`, prediction seed `4.501354368358079e-06`, logvar gradient `7.711420010991787e-06`. Standalone mixed absolute/relative bounds track magnitude: square1e-7+2e-7*abs(ref), loss1e-6+3e-7*abs(ref), seed1e-7+2e-7*abs(ref), logvar2e-7+3e-7*abs(ref). A separate smooth F64 finite difference validates both prediction/logvar analytical gradients within1e-9. Zero residual and boundary logvar±80 validate signed loss and complete zero seeds.

These measure F32 arithmetic/reduction rounding in a host interpreter using F64 exp rounded toF32, not native transcendental error. Pinned full-step loss3e-5, all-gradient/input5e-5 and update/EMA4e-6 tolerances remain unchanged. Nine differential complete steps cover EOS weights0/.1/1 and diagonal weights0/.75/1, including zero-scale branches. Independent fixtures and legacy tests check endpoint stop-gradient and mixed-AD regressions. No new audio/listening acceptance exists.

Both named scopes pass checkpoint/resume parameters/M/V/EMA/raw replay with stable resident allocation count. Every tiny complete-step tape admission position is injected; unchanged host/replay, no retry/checkpoint, retained owner. Tests cover malformed names/topology/budgets/nonfinite/overflow, wrong/inactive loss owner and legacy injection. Offline PTX ABI and Vulkan binding/contract mocks pass.

A mathematical delegate confirmed residual VJP/stop-gradient and scaling if applied once, and noted that conventional log-variance would use opposite signs. The implementation deliberately preserves the pinned precision-style upstream contract. This was formula review only, not source/hardware approval.

## Final offline gate

**00:36:48.185702750--00:37:51.433015981 UTC**, exit0. Thirteen tests repeated ten times gave **130 top-level+110 subtest passes**; race **13+11**; affected regressions **41+104**. No failures/skips.

Local Go1.26.3, GOENV off/empty GOFLAGS, GOPROXY/GOSUMDB off, GOMAXPROCS2, both GPU-disable flags, nice10, serial packages/tests,30-second inner/120-second outer deadlines. Affected backend/model/CLI vet, host build, ARM64/RISC-V CLI cross-builds, layout and whitespace gates pass. Foreign binaries not executed. Offline CUDA12.8 nvcc/PTX/sm80 ptxas plus GLSL/SPIR-V validation pass; no new SPIR-V opcode/extended instruction required. Initial lexical `<-80` parse issue fixed without numerical changes.

Scoped coverage: PTX/Vulkan primitives100%; model builder/boundary/scope/owner100%; loss evaluator88.2%; total selected package19.0%. Native execution untested. No released-size fit, speedup or voice-quality claim.

Evidence: `/workspace/exports/pocket-gpu-training-loss-20261002/`. Remaining: device interpolation/branch orchestration and all-device global norm; reduce dense embedding VJP, graph/activation churn, CPU transfer boundaries/full snapshots. Native trig/loss/released-size numerics/resources/performance and sampler/new-voice/listening qualification await coordination.
