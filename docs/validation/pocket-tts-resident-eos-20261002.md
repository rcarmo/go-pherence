## Pocket TTS resident EOS loss -- 2 October 2026

`ptx-resident-eos-hybrid` and `vulkan-resident-eos-hybrid` add stable EOS binary cross-entropy, weighted logit seeds and a compensated scalar mean reduction to the resident-time path. All trainable parameter gradients remain resident. LSD scalar loss/flow orchestration, final global norm-summary reduction, tape transfer boundaries and completed-update full host snapshots remain. This is not full-device training.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Training source is uncommitted; staged inference index unchanged at eight paths /469 insertions. No GPU/runtime initialisation, probe, execution, reset/recovery, real model, audio or listening run occurred. Execution and inference listening/publication holds remain.

## EOS arithmetic and policy

The tape labels each position as -1 ignored,0 negative target,1 EOS target. Enabled positions are `mask[0]` at frame0 and `mask[i-1]` thereafter. Only the first invalid frame after a contiguous valid prefix can be an enabled positive target; all later padding is ignored. Frame0 is never EOS. All-invalid/empty/non-prefix masks reject before execution. Full-valid truncated windows contain no positive target, matching the CPU policy. Ignored nonfinite logits are sanitised to0; active logits must be finite.

For `e=exp(-abs(x))`, loss is `max(x,0)+log1p(e)` for target0 and `max(-x,0)+log1p(e)` for target1. Stable gradients use signed residuals directly rather than subtracting1 from an F32 sigmoid. Positive-target x>=0 uses `-e/(1+e)`, preserving small positive-logit tails. Gradient is divided by active count then multiplied by EOSLossWeight. Loss is unweighted, matching metrics before the host final objective combination. A single thread uses compensated F32 summation and mean; no float atomics. Limits are1..1024 logits/positions. Active logits are bounded by `(MaxFloat32/2)/activeCount` to leave reduction/compensation headroom; CPU F64 legacy loss remains wider.

PTX uses expf/log1pf with no fast math/FMA contraction. GLSL uses Exp and Log; for e<.125 a12-term alternating log1p series avoids dropping small softplus tails when1+e rounds to1. The new unary GLSL.std.450 Log is admitted; a mutated Log2 still rejects. Offline SPIR-V framing is metadata admission, not full safety/numerical proof.

`ResidentState.RunUnboundTape` runs loss/utility tapes with the existing complete tape admission, active borrow guard, output validation and terminal retained-owner rules. It cannot bind or accumulate parameters; no fake parameter/gradient placeholder is needed. The original RunTape API still rejects empty bindings. The EOS backend verifies scope, model/component owner and active prepared step; bad results poison its owner. Loss failure prevents host update, replay commit, retry or checkpoint publication.

New CLI/replay identities `*-resident-eos-hybrid-v1` preserve prior F64 loss arithmetic/scopes. The tiny full-step fixture downloads104 boundary tensors (102 prior + scalar EOS/seed), with no parameter-gradient download/upload at update. This is a tensor count, not transfer bytes or latency. EOS logits/seeds still transfer through CPU between projection and loss tapes. Resident/tape ceilings remain separate conservative logical bounds, not total native/host memory.

## Offline numerical checks

Independent CPU F64 reference:30 cases across lengths1/2/7/257/1024, valid-prefix boundaries and weights0/.1/1, with logits -100..100. Maximum mean-loss error `1.2351889857598053e-06`; maximum weighted logit-gradient error `2.3283064365386963e-10`. Standalone loss bound2e-6+1e-7*abs(reference), gradient6e-8 is calibrated to F32 element/reduction rounding; downstream fixture bounds are unchanged. Independent stable F64 target-one tails through x80 give maximum relative error `4.040891121478698e-08` (bound2e-7). The GLSL log1p series has maximum absolute error `1.9365591602538146e-08` across10,001 points (bound2e-8). Host interpreter uses F64 transcendental calls rounded toF32; this isolates graph/rounding but does not measure native exp/log error or underflow.

Pinned complete-step PyTorch comparisons remain loss3e-5, all gradients/input5e-5 and parameter/EMA updates4e-6. Nine differential full-step variants cover EOS weights0/.1/1 and diagonal-objective weights0/.75/1, checking every gradient/update/EMA with the same bounds. Both named scopes pass checkpoint/resume parameters/M/V/EMA/replay, stable residency and cross-scope rejection. Every tape-construction position in a complete step is injected; host/replay stays unchanged and failed owner is retained. Unbound coordinator tests cover construction/nil/run/download/count/extent/nonfinite/close failures, borrowing and non-mutation of optimiser state.

A self-contained mathematical delegate found the stated stable loss/gradient formulas correct, conditional on later padding being inactive and zero active count rejecting. Both conditions are explicitly enforced and tested. It did not review source or hardware.

## Final gate

**00:24:35.055557275--00:25:36.824945481 UTC**, exit0. Fifteen tests repeated ten times: **150 top-level +190 subtest passes**. Race: **15 +19**. Affected regressions: **36 +92**. No failures/skips.

Controls: local Go1.26.3, GOENV off, empty GOFLAGS, GOPROXY/GOSUMDB off, GOMAXPROCS2, both GPU-disable flags, nice10, serial packages/tests,30-second inner/120-second outer deadlines. Affected backend/model/CLI vet, host build, ARM64/RISC-V CLI cross-builds, layout (five Bun tests/3008 assertions plus Go docs) and diff whitespace checks pass. Foreign binaries not executed. CUDA12.8 nvcc/PTX/sm80 ptxas and GLSL/SPIR-V offline validation pass. Initial shader reserved-field/framing issues corrected without widening the supported opcode envelope beyond unary Log or changing numerical tolerances.

Scoped coverage: PTX EOS100%, Vulkan EOSStage100%, model builder/scope/owner/boundary100%, EOS evaluator92.9%; total selected package statements19.6%. Native execution remains untested. No measured released-size fit, performance gain, listening or trained-voice acceptance.

Evidence: `/workspace/exports/pocket-gpu-training-eos-20261002/`. Remaining: device LSD scalar loss/flow orchestration and all-device global norm; reduce dense embedding VJP, graph/activation churn, inter-tape CPU transfers and full snapshots. Native time-trig/loss/released-size numerics/resources/performance and sampler/new-voice/listening qualification await coordination.
