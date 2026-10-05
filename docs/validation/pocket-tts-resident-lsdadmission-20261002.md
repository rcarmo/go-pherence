## Scalar LSD prediction admission primitive

LSD-loss mode two now reports a scalar invalid flag for prediction/desired finiteness and the existing conservative difference/overflow bound. It computes no loss or gradients and does not accumulate parameters. Current Pocket TTS builders, CLI choices and replay scopes do not use it; prediction downloads and host admission remain unchanged until a separate integration is qualified.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and469insertions. All execution used offline CPU/mock qualification with NVIDIA/Vulkan disabled. CUDA/PTX and GLSL/SPIR-V compilation/inspection were offline only. No native GPU, released-model/audio, recovery or publication work occurred.

## Preserving the original guard

The original host admission computes `d = float32(prediction-desired)`, checks prediction/desired/difference finiteness and compares the difference magnitude in F64 against both:

- `sqrt((MaxFloat32/4)/channels/max(1,exp(logvar)))`;
- the doubled seed ceiling `2*abs(d)*exp(logvar)/channels <= MaxFloat32/4`.

Channels are1--16,384 and logvar is finite in[-80,80]. These are conservative admission conditions, not the loss objective or a reference derivative.

`LSDPredictionCutoff` evaluates exactly those F64 comparisons on the host and binary-searches the monotonic positive-finite F32 bit range for the largest admitted magnitude. It does not approximate a combined real-valued threshold and round it twice. Since the runtime difference is F32, checking its magnitude against that maximal admitted F32 value preserves the same comparison boundary for finite inputs, including equality.

Mode two receives that cutoff in the third input, in place of logvar, and carries the provenance logvar in command Scale. Tape admission recomputes the cutoff and requires the explicit constant to match it. This exact constant comparison is an admission contract, not generated-signal equivalence. Invalid dimensions/logvar, missing or changed cutoff constants, malformed shapes, dead nonfinite source tensors and full logical byte ceilings are rejected before execution.

The kernel uses one thread to scan the complete row, setting output to1 if either input is nonfinite or the F32 difference magnitude exceeds the cutoff, otherwise0. It never squares inputs, computes exp, changes parameters or produces gradient seeds. Large identical finite inputs remain admissible because their difference is zero. NaN/Inf and finite subtraction overflow fail. CUDA uses `--fmad=false` without fast math; GLSL marks the difference precise. Native FTZ/subnormal/runtime rounding remains unqualified.

Raw callers must supply admitted cutoff contents, just as raw loss callers must supply valid logvar/headroom inputs. Runtime wrappers validate output/input extents and owners, not device-buffer contents. The operation reuses the existing LSD-loss ABI, pipeline and PTX entry, with one scalar output; no extra pipeline or kernel entry is added. Modes zero/one retain their original output extents, arithmetic and current routing.

## Offline boundary evidence

Independent test code reconstructs the original host F64 comparisons rather than deriving expectations from the cutoff helper. Across channels1,2,16,1,024,16,384 and logvar-80,-1,0,.1,1,20,80, the cutoff passes and its next representable F32 magnitude fails. Neighboring positive/negative magnitudes, zero, maximum finite, Inf and NaN have matching predicates. These are Boolean boundary tests for the admission contract, not an error-tolerance widening or numerical speed claim.

Pocket TTS interpreter tests compare scalar flags against the existing `admitLSDLossTraining` for small and maximum-channel graphs, both cutoff sides/signs and pruning. Additional cases cover large identical inputs, opposing half-cutoffs, rounded F32 subtraction, neighboring halves and subnormal differences. Finite source constants generate Inf/NaN at runtime through ordinary vector commands to exercise the scalar flag without bypassing full-source validation.

An initial runtime-invalid test incorrectly supplied nonfinite constants and was rejected by the interpreter's complete-source admission. It was replaced by finite-source runtime generation. Initial CUDA compilation caught incorrect source variable names; those were corrected before final compilation. Development failures remain archived and are not current blockers.

PTX mock tests cover modes zero/one/two, the unchanged six-argument ABI, scalar dispatch, pointer order, owner/extents/alias errors, missing entry, terminal launch/sync poisoning, retry rejection and preflight. Vulkan tests cover modes zero/one/two, scalar stage/tape dispatch and the unchanged four-binding/eight-push-byte/local-size contract, invalid mode/owner/extents/alias, cancellation and preflight. Conservative shader reflection and `spirv-val` pass without widening the opcode envelope. Formula/interpreter tests do not numerically execute compiled kernels.

## Offline gates

Evidence directory: `/workspace/exports/pocket-gpu-training-lsdadmission-20261002`.
Gate script: `/workspace/tmp/pocket-lsdadmission-gate.sh`.
Final window: **2026-10-02 11:13:54--11:15:08 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions |100 top-level|0|
| Selected tests, race detector |10 top-level|0|
| Affected CPU/mock regressions |775 top-level +1,277subtests|1|
| Compiled-artifact follow-up |10 top-level|0|

Zero failures in final gates. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

Cutoff helper, LSD shape and both runtime LSD wrappers have100% scoped statement coverage. Package coverage is89.4% training,12.9% NVIDIA runtime,36.7% Vulkan,68.8% Pocket TTS and14.9% pockettrain under this selector. Existing pinned PyTorch/CPU/F64 gradients, updates, clipping, replay and retention tests pass unchanged numerical gates. No generated-output hashes or byte identity are used as signal acceptance gates.

Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass. Offline CUDA12.8 sm80/PTX8.7 compilation/ptxas and GLSL compilation/spirv-val/reflection pass. These do not establish native correctness, resource use, latency or voice quality.

## Review and next work

The invariant judge timed out after15seconds; no independent source/native approval is claimed. Integration must put this read-only scalar check before backward accumulation, keep all source admission and required row values alive, preserve legacy/public results and define a distinct scope if downloads/guard orchestration change. This primitive alone removes no downloads and releases no hold.

Transformer/flow handoffs, full prediction/device-boundary integration, EOS diagnostics, dense persistent gradients/optimiser work and snapshots remain. Native resources/performance and new-voice/listening acceptance are held. The overall implementation goal remains active.
