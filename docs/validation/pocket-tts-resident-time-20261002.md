## Pocket TTS resident time embeddings -- 2 October 2026

`ptx-resident-time-hybrid` and `vulkan-resident-time-hybrid` compute sinusoidal time values, selected-time JVPs and reverse-over-forward time adjoints inside flow tapes. All trainable parameter gradients remain resident. Scalar loss/flow orchestration and final global norm-summary reduction remain on CPU; tape boundaries still transfer through CPU and complete host snapshots synchronise each update. Full-device training is unfinished.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. New training work is uncommitted; held inference index unchanged at eight paths / 469 insertions. No GPU initialisation, probe, launch, recovery, real-model, audio or listening run occurred. Execution and inference publication/listening holds remain.

## Sinusoidal operations

`time-embedding` has four F32 inputs: frequencies `[half]`, `[time,dTime]` and two adjoint vectors `[2*half]`. Mode0 produces `[cos(time*f), sin(time*f)]`; mode1 produces `[(-sin(time*f)*f)*dTime, (cos(time*f)*f)*dTime]`. Mode2 computes the scalar adjoint to time in frequency order:

```
fc = -sin(angle)*f; fs = cos(angle)*f
sc = -cos(angle)*f*f; ss = -sin(angle)*f*f
sum += seedValueCos*fc + seedValueSin*fs
sum += seedTangentCos*sc*dTime + seedTangentSin*ss*dTime
```

The selected-time tangent is fixed to zero or one, not differentiated. Frequencies are immutable checkpoint inputs, not trainable parameters. Bounds: `1<=half<=8192`, exact ranks/extents, finite inputs, finite squared frequencies/derivative scale and `abs(time*f)<=8192`. Raw kernels require the same finite/bounded contract; tape admission checks explicit frequency/time constants. F32 multiplication rounds the angle before native trigonometry. Destination storage is disjoint from all inputs; seed inputs may share read-only zero storage. One scalar thread owns the reverse reduction without atomics.

PTX uses `sinf`/`cosf` with no fast math/FMA contraction; Vulkan uses GLSL sin/cos with precise arithmetic around the operations. SPIR-V contract inspection adds only unary GLSL.std.450 Sin/Cos, alongside existing permitted functions. A mutated Tan instruction is rejected in the offline shader test. Shader contract framing remains metadata admission, not semantic proof or native numerical acceptance.

`BuildFlowDeviceTimeTrainingTape` is a separate builder; `BuildFlowDualTrainingTape` retains host F64-rounded trigonometry and its legacy boundary ordering. Resident-time scopes inherit weighting/projection/sequence ownership and select the new builder. Returned time boundaries are one scalar per time input rather than two embedding vectors. Pinned tiny complete step downloads **102** boundary tensors versus126 in resident-sequence; this counts tensors, not bytes/latency. No parameter gradients download or upload at update. New replay scopes are `*-resident-time-hybrid-v1`; old bundles reject cross-scope restore before mutation.

## Numerical evidence and limitations

Independent F64 calculus covers14 cases (seven times, tangent0/1, five frequencies), through angles +/-8192. The host interpreter uses F64 sin/cos rounded to F32 to isolate graph/reduction arithmetic. Measured maximum errors against independent F64: value `2.9256811640010483e-08`, tangent `7.454474001633571e-06`, scalar time adjoint `0.000715458834747551`. This is F32 term/reduction rounding, **not native trig error**. The standalone bounds are6e-8 for values,2e-5 for scaled tangents and .002+2e-7*abs(reference) for large-frequency scalar adjoints. A separate smooth F64 finite difference validates the analytical second-derivative formula within2e-8. These standalone scales do not widen downstream fixture tolerances.

Pinned independent PyTorch mixed-AD fixture stays at output3e-6, tangent5e-6, input/condition/time8e-6 and parameters1e-5. Full-step fixture stays at all-gradients5e-5, loss3e-5, parameter/EMA update4e-6. Complete tiny update still has zero host parameter-gradient uploads. Unselected time tangents are zero; time-gradient boundary malformed/finite checks and frequency/angle/budget rejection are tested.

Native sin/cos may differ from the interpreter. Derivative error scales with frequency and frequency squared; high-frequency tests and complete gradients/loss/update/quality must be measured on each supported released device before numerical acceptance. No native accuracy, resource fit, speedup, trained-voice quality or listening claim follows from this offline result.

## Validation

Final gate: **00:09:09.816001840--00:10:11.641241911 UTC**, exit0. Twelve tests repeated ten times gave **120 top-level +20 subtest passes**; race gave **12 +2**. Affected regressions gave **41 +83**. No failures or skips. Both named scopes pass checkpoint/resume with parameters/M/V/EMA/replay and stable resident allocation count. Every tape-admission failure position in a tiny step is injected; host/replay unchanged, retry/checkpoint boundary unavailable, resources retained. Explicit changed component owners reject.

Controls: local Go1.26.3, GOENV off, empty GOFLAGS, GOPROXY/GOSUMDB off, GOMAXPROCS2, both GPU-disable flags, nice10, serial packages/tests,30-second inner/120-second outer deadlines. Affected backend/model/CLI vet, whole-tree host build, ARM64/RISC-V CLI cross-builds, layout and whitespace checks pass. Foreign binaries were not executed. CUDA12.8 nvcc/PTX/sm80 ptxas and GLSL/SPIR-V offline validation pass. Initial test fixture-field and shader-select framing errors were fixed without relaxing numerical gates.

Scoped coverage: PTX TimeEmbedding100%, shape admission100%, Vulkan stage94.1%, flow tape builder95.5%, boundary sizing100%. Selected package total14.9%; native execution coverage remains zero. A self-contained mathematical delegate confirmed the reverse-over-forward formula for fixed dTime and found no formula bug; it did not review source or hardware.

Evidence: `/workspace/exports/pocket-gpu-training-time-20261002/`. Remaining implementation: device scalar loss/flow orchestration and all-device norm; dense embedding VJP, graph/activation churn, inter-tape transfers and full snapshots remain baseline costs. Native released-size numerical/resource/performance, sampler/new-voice/listening qualification awaits coordination.
