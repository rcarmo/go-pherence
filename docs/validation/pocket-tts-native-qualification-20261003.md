# Pocket TTS native qualification, 3 October 2026

The tiny pinned full-step fixture now passes actual PTX and Vulkan execution, including device G, AdamW P/M/V and EMA. The released-model SIMD command also completes preparation, updates, checkpoint/resume and inference export on a labelled synthetic recording. Released-size resident GPU execution is still blocked by the current memory layout.

Rui cleared production/native qualification at 14:56 UTC, then required caution with PTX and an audit before every run. No other sessions were contacted or delegated to. PTX source and kernel assets were not changed. Existing admission limits, numerical tolerances and staged inference eight files/469 insertions remain intact. Source parent is `a90aec30567d485784f2e04d1803c543c9f2817f`, with uncommitted training work. Binary/source manifests bind the working tree separately.

## Tiny native numerical checks

`TestPocketNativeFullStepPyTorch` is explicitly opt-in through `GO_PHERENCE_POCKET_NATIVE_FULLSTEP=ptx|vulkan`. It uses the existing schema-1 fixture pinned to upstream `0acce6b2f390150267557770d2098c5caa9a18ac`, 3 frames/2 active, hidden4/latent2. Frozen P/buffers/batch/samples/config are admitted before runtime initialisation. It selects `*-resident-devicehandoff-hybrid-v1` with unchanged 1MiB resident/tape limits for this fixture. There is no fallback.

The test-only driver captures each completed device gradient before its update, then checks independent fixture G/P/EMA, final derivatives and moments derived from the independent G. All five metrics use the existing3e-5 gate; derivatives5e-5; G8e-5; P/EMA/M/V4e-6. Device-owned host G placeholders remain zero. Snapshot and host completed-boundary checks precede normal close.

Hardware: RTX3060 12GiB, UUID `GPU-cb9db783-7465-139d-900c-bc8d6b6ced7c`, driver610.57.04, Go1.26.3 linux/amd64, GOMAXPROCS2. Health checks reported0% load/33MiB baseline,53–55°C; kernel journal search found no new NVRM/Xid lines in the inspected window. Each test ran in a fresh process, sequentially, with the other GPU backend disabled.

| Quantity | Values | PTX max absolute error | Vulkan max absolute error |
|---|---:|---:|---:|
| Final derivatives | 14 | 2.98023224e-8 | 2.98023224e-8 |
| Device G | 606 | 2.98023224e-7 | 2.98023224e-7 |
| Updated P | 606 | 1.49011612e-8 | 1.49011612e-8 |
| EMA | 606 | 1.1920929e-7 | 1.1920929e-7 |
| M/V | 1,212 | 2.98023224e-8 | 2.98023224e-8 |

PTX completed-update diagnostic interval44.34ms; Vulkan817.31ms. These single tiny runs include device-G diagnostic capture and exclude startup. They are correctness evidence; no released-model efficiency comparison follows.

The first PTX process used the wrong working directory and failed to find the fixture after runtime startup; it performed no training. The test now admits the fixture before runtime startup. A corrected package-directory run passed. PTX has not been rerun after the stricter per-run audit instruction.

## Vulkan norm defect and correction

The initial Vulkan update rejected `invalid/nonfinite global norm boundary` before optimiser mutation. Diagnostics isolated a positive chunk maximum `.022259595` with summary `[maximum,0.9999999,0]`. The unchanged merge requires a nonzero chunk's scaled squared sum to be at least1. Native division lowered `x/x` to a value below1.

The norm-stat shader now assigns exact1 to each element whose absolute value equals the computed maximum. The global merge assigns exact1 to a chunk/global equal-maximum ratio. Squaring makes the element's sign irrelevant. Nonmaximum divisions, compensated sums, invalid flags, bounds, clip guards and host gates are unchanged. Both SPIR-V assets were rebuilt with `glslangValidator -V --target-env vulkan1.1 -S comp` and checked by `spirv-val`. PTX shaders were untouched.

`TestVulkanNativeNormUnitMaximum` covers positive/negative singleton maxima in lengths1/3/255/256/257/513 and a513-value mixed-magnitude F64 reference. Singleton norm relative error is0. Mixed norm relative error is1.70881785e-8, within the existing3e-7 gate. Independent clipping checks also pass. The audited final Vulkan full-step passes after the correction. Diagnostic source hooks were removed; failed runs/logs and prior SPIR-V assets remain preserved.

Affected offline shader-contract/admission/retention checks:30 top-level repeated passes,3 race passes, zero failures/skips. Full-step and norm tests remain native opt-ins; ordinary test execution does not initialise the device.

## Released CPU command

The actual `pockettrain` executable uses:

- Released voice-enabled model `fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997`,219029196 bytes, already local.
- Tokenizer `f498428e1eafee50492f7be13dc9bfafcfc12e508cd0eb1b01c92ecd5d8c6687`,245020 bytes.
- Repository `model/pockettts/testdata/english-upstream.yaml`, SHA256 `1b06236b6a4405a4010c731da097745c23d3d5c543e712e31086b6293fd1124a`.
- Synthetic220Hz tone,49920 PCM16 samples/2.08s, mono24kHz, amplitude≤1800/32768. Parser-fixture words `test voice`, explicit split1.04s. The words do not describe speech ground truth.

Preparation actually runs frozen Mimi on both13-frame segments, normalises the target and adds one EOS row:14 physical/13 valid target frames,13 voice frames,one token. The prepared corpus has real model/config/tokenizer/manifest/policy provenance; no fabricated prepared inputs bypass the command. Both GPU backends are disabled. Default6GiB host and2GiB checkpoint ceilings stay unchanged. This recording supplies engineering coverage and no voice-quality evidence.

| Command | Wall time | Peak RSS, KiB | Result |
|---|---:|---:|---|
| Prepare | 0.61s | 144,172 | One prepared row |
| Train1 + checkpoint | 7.37s | 3,381,504 | Committed step1 |
| Resume + train1 + checkpoint + EMA export | 11.47s | 4,665,728 | Committed step2 and fresh export |
| Continuous train2 + one final checkpoint | 8.59s | 3,383,168 | Committed step2 |

These intervals include process loading and checkpoint work; they are not steady-state seconds/update. Each model bundle is1431217112 bytes. Old immutable generations remain. The EMA export is438033312 bytes with214 finite F32 tensors:127 FlowLM and87 frozen Mimi; training-only weighting tensors are absent. Its structure/coverage was independently scanned; inference/listening and independent upstream reload have not run.

A separate streaming Bun decoder compares all completed step2 P, fixed buffers, M/V and EMA between continuous and checkpoint/resumed commands. It uses numerical4e-6 acceptance, finite/nonnegativeV checks, exact name/extents and step/config/replay association. Checkpoint footer hashes verify file integrity only. Random generation filenames differ and are validated separately from replay equality.

| State | Names | Values | Mean/RMS/max absolute error |
|---|---:|---:|---:|
| P | 131 | 89,449,730 | 0/0/0 |
| Fixed buffers | 4 | 320 | 0/0/0 |
| M/V | 131 | 178,899,460 | 0/0/0 |
| EMA | 131 | 89,449,730 | 0/0/0 |

Replay RNG, permutation, cursor, epoch and numerical identity agree at step2. This is one short synthetic-row restart qualification; longer rows, cancellation and sustained resource reliability remain open.

## Corrected resource inventory

The actual CPU state inspection exposed an earlier synthetic-audit error. It counted the conditioner embedding as4000×1024 and built a matching4000-row synthetic model. Released training uses4001 rows, including the padding token. Both synthetic inventory/model now include that row and cross-check total elements against the independent allocation-free `trainingParameterElements` planner.

| Quantity | Corrected bytes/elements |
|---|---:|
| Trainable elements | 89,449,730 |
| F32 P | 357,798,920 bytes |
| Legacy eight-buffer resident + statistics/globals | 2,866,584,460 bytes |
| Excess above2GiB | 719,100,812 bytes |
| Sharedworkspace resident plan | 1,839,522,904 bytes |
| Shared resident-only headroom below2GiB | 307,960,744 bytes |
| Shared + known complete distillation source lower bound | 2,315,561,792 bytes |

The old89,448,706/2,866,551,644/1,839,502,424 figures in historical evidence undercount by1024 elements. Existing historical logs are preserved; these corrected figures supersede their released arithmetic. Maximum parameter extent4194304 and summary49152 remain unchanged. Legacy admission still rejects before native allocation/upload. The shared native adapter/CLI route does not yet exist.

Focused corrected-resource gate:40 repeated top-level passes and4 race passes, zero failures/skips. Required vet, formatting, SPIR-V validation and diff checks pass. No untouched broad/component gates were repeated.

## Audit and remaining work

Pre-run records and source manifests are under `/workspace/exports/pocket-native-qualification-20261003`. After Rui's per-run audit instruction, the Vulkan norm regression, final Vulkan full-step, CPU preparation/update/resume/continuous commands and resource gates each have recorded local source/bounds/alias/ownership/launch/failure audits. No other agent was asked to review. These records are local checks; independent review has not run.

Next: guarded native sharedworkspace adapters and a distinct runner/backend identity; complete working-set admission before released GPU allocation; larger-shape native differential and failure checks; released short/long backend timing and peak-memory matrix; integrated cancellation/soak/export loadability. Audit the actual PTX source/embedded kernels and launch path before every PTX run; do not change PTX opportunistically or run larger shapes before resource admission. Stop on device uncertainty, retain owners and perform no automatic reset/retry. Publication and staged-inference listening holds remain.
