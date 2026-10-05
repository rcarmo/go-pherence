## Pocket TTS device-global norm/clipping -- 2 October 2026

`ptx-resident-globalnorm-hybrid` and `vulkan-resident-globalnorm-hybrid` compute global gradient norm reduction and clipping scale on device, alongside dependency-pruned training tapes and resident AdamW/EMA. Four diagnostic/scale scalars transfer before update; post-update mutable-state summaries and completed-step snapshots still transfer. Host dispatch, boundary/weighting-input assembly and AdamW bias constants remain. Full-device training is unfinished.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Source uncommitted; staged inference index unchanged at eight paths/469 insertions. No GPU/runtime initialisation/probe/launch/recovery, real-model, audio or listening run occurred. Execution/listening/publication holds remain.

## Two-pass scaled norm

Existing <=256-value chunk statistics contain `[maximum, scaledSquare, invalid]`. Finite gradients avoid direct square overflow by summing `(gradient/chunkMaximum)^2`, with compensated F32 arithmetic. The device-global path reads these summaries twice:

1. Validate all chunk flags/extents and find fixed global maximum M. Accumulator square/correction stay zero.
2. Sum `(chunkMaximum/M)^2 * chunkScaledSquare` with F32 compensation that persists across every parameter boundary. M stays fixed. Zero M produces zero square.

The summary ping-pong buffers hold `[M,S,invalid,correction]`; the clip output replaces correction with scale. Global norm is represented by M*sqrt(S). Clipping computes root=sqrt(S), ratio=maxNorm/M, and scale=ratio/root when ratio<root, else1. Zero M uses scale1. This avoids squaring M or materialising the norm in F32, including when the true norm exceeds MaxFloat32. Counts are bounded to2^38; S is bounded to total elements with a2e-6 rounding envelope. Invalid flags propagate and reject before AdamW mutation. Kahan arithmetic requires ordered operations; CUDA disables FMA contraction/fast math and GLSL uses precise arithmetic around reduction terms.

The coordinator admits the entire max/sum/clip pass before dispatch, constructs chunk summaries only once, then dispatches both summary passes. Twelve owned F32 scalars add48 bytes to resident accounting. Errors during allocation/launch/download retain owners; pure metadata failure does not dispatch or mutate model state. Every parameter must have a ready gradient before reduction. Missing gradients are not implicit zeros.

`NewGlobalNormResidentState` opts into this path; `NewResidentState` retains host F64 reduction unchanged. New scopes are `*-resident-globalnorm-hybrid-v1`; older CLI/replay paths reject cross-scope restore. Native adapters verify exact disjoint four-scalar accumulators, chunk extents, modes0/2, counts and kernel availability. PTX launch/sync failures poison the operator; Vulkan submission/close failures retain plans.

The final four scalars `[M,S,flag,scale]` transfer for validation and diagnostic F64 `M*sqrt(S)`. Host code does not recompute the clipping decision; the device scale is passed unchanged through existing AdamW launch constants. Bias corrections still use host Pow. Post-update parameter/M/V/EMA checks still download per-parameter summaries, and coherent snapshots still download complete state. The two-parameter fixture changes gradient-summary downloads from2 to1, while retaining8 post-update summary downloads. More merge dispatches and a second summary scan add overhead; native latency/memory and speedup are unmeasured.

## Drift fix and independent evidence

The first design repeatedly rescaled the accumulated square whenever a larger parameter maximum appeared. A new4096-parameter slowly increasing-maximum test found norm relative error `1.0039249192772721e-05` and scale error `3.74360215060765e-06`, violating the existing3e-7 norm gate. The failed log is preserved. Fixed maximum plus persistent compensation reduces the same case to norm relative error `5.058707927602878e-09`, scale absolute error `1.1490470241959372e-08`. No tolerance was widened.

Independent F64 hypot over gradients checks magnitudes0/1e-30/1e-10/1/1e10/1e30, vectors1/257/4097 and four ceilings. Maximum norm relative error `2.4753937702171775e-08`; scale absolute error `6.390902780942231e-11`. Bounds remain norm3e-7 and scale1e-7+3e-7*reference. Additional4096-parameter flat/increasing/mixed patterns span65482 elements; all pass unchanged bounds. MaxFloat32 pairs verify overflow-safe diagnostics and nonzero representable clipping scale. Native flush-to-zero/subnormal behavior is unmeasured and can drop tiny contributions or scales; this requires device qualification.

Pinned complete-step metrics/loss3e-5, all-gradient/input5e-5, updates/EMA4e-6 remain unchanged. Three clipped successive updates use independent CPU gradients and F64 direct norms, at ceilings.01/.1/1e20; parameters/M/V/EMA compare within unchanged bounds. Nine objective variants, padded rows, both named checkpoint/resume scopes and replay identity reject/pass as expected. Invalid gradient summaries block all updates; tests inject max/sum preflight, sum execution, clip, stats/download/update and last-three-buffer allocation failures. Failed steps do not advance replay/checkpoint boundaries and retain uncertain resources.

A formula delegate reviewed both prototypes; the corrected fixed-max formula was sound subject to explicit zero-max handling, finite maxNorm, strict ordering and native subnormal limits. Review covered stated mathematics only, without final source/hardware approval.

## Final offline gate

**01:29:13.053077190--01:30:26.804223679 UTC**, exit0. Twenty-one tests repeated ten times: **210 top-level+180 subtest passes**; race **21+18**; regressions **114+196**. No failures/skips. Earlier single-pass gates are retained as historical evidence and do not qualify the final revision.

Local Go1.26.3, GOENV off/empty GOFLAGS, GOPROXY/GOSUMDB off, GOMAXPROCS2, both GPU-disable flags, nice10, serial packages/tests,30-second inner/120-second outer deadlines. Affected backend/model/CLI vet, host build, ARM64/RISC-V CLI cross-builds, layout and whitespace gates pass; foreign binaries not executed. CUDA12.8 nvcc/PTX/sm80 ptxas and GLSL/SPIR-V offline validation pass. GLSL conditionals were rewritten to fit existing opcode framing without extending it.

Scoped shared reduction94%+, PTX execution88.9%/admission95%+, Vulkan stage95%+, constructor100%; selected package21.8%. Native execution coverage absent. Evidence: `/workspace/exports/pocket-gpu-training-globalnorm-20261002/`, with final files named `*-two-pass`.

Remaining: host-free norm-to-AdamW handoff, fused inter-tape/weighting-input orchestration, host graph/constants/live-allocation churn and complete snapshot traffic. Native trig/loss/norm/released-size numerical/resource/performance and sampler/new-voice/listening qualification await coordination.
