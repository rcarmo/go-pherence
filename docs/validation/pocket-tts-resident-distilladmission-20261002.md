## Resident distillation prediction admission

Two opt-in scopes now add scalar admission to the private minimal-LSD forward tape:

- `ptx-resident-distilladmission-hybrid-v1`
- `vulkan-resident-distilladmission-hybrid-v1`

They inherit diagonal scalar admission and compact embedding. Predictions and all six distillation forward vectors still download; the original host guard remains redundant. This adds read-only admission before endpoint/primary backward, not a transfer or speed improvement.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths/469 insertions. Execution was offline CPU/mock with NVIDIA/Vulkan disabled; CUDA/PTX and GLSL/SPIR-V compilation was offline only. No native/model/audio/recovery/publication work occurred.

## Private policy and ordering

The private step passes known log variance through `trainingDistillAdmissionBackend` only when `stepOnly` and fused forward are enabled. Public minimal-LSD calls and the public forward interface retain their original routes/results. The new capability chain also keeps the preceding diagonal guard.

`appendDistillAdmission` validates the entire source tape and resident binding inventory, including unused tensors, before copying metadata. It appends a cutoff constant and mode-two LSD command reading prediction output5 and desired output4. The maximal-F32 cutoff retains the original F64 square/seed-overflow rule from [scalar primitive validation](pocket-tts-resident-lsdadmission-20261002.md). The standalone loss byte ceiling and expanded graph/binding ceilings both apply before pruning/allocation.

The tape executes with accumulation disabled. The private call downloads the original six vectors plus the scalar. A finite-zero flag with correct count/shape and no sibling/caller alias is required; then the original six-output decoder validates vector dimensions, finite values and ownership. Original host `admitLSDLossTraining` rechecks prediction/desired/logvar. Only successful return reaches endpoint/primary VJP and primary parameter accumulation.

Tensor IDs, original commands/output slots, arithmetic, stop-gradient, parameter scale and exactly-once roots are unchanged. Appended cutoff/operand/output metadata is owned; original nested metadata remains read-only borrowed. Public LSD policy is tested separately: it requests no new distillation scalar, while private step policy requests one.

This is a per-row guard, not gradient-atomic rollback for the whole step. Earlier successful rows may have accumulated resident gradients when a later row fails. The runner/backend then become terminal: no optimiser update, completed state, metrics/replay publication, checkpoint or retry is allowed. This distinction is explicit rather than claiming the accumulation buffer stayed empty.

## Scope and transfer cost

CLI names are `ptx-resident-distilladmission-hybrid` and `vulkan-resident-distilladmission-hybrid`. Opt-in, runtime pin and logical ceilings remain required. Factory/component owner tests cover both new scopes. Hash kind is `resident-distilladmission-backend`; same-scope resume passes, and earlier diagonal/pruned scope restores reject without mutation. Existing hashes/routes remain unchanged.

Tiny CPU/mock complete-step accounting versus diagonal-only admission:

| Measure | Diagonal-only baseline | Diagonal + distillation |
| --- | ---: | ---: |
| Tapes |11|11|
| Commands |3,711|3,713|
| Downloads |44|46|
| Downloaded elements |119|121|
| Executed logical bytes |134,044|134,060|
| Planned storage bytes |41,964|41,980|

Two valid distillation rows add one command, scalar download and eight logical bytes each. Accumulation counts are unchanged. Compared with the compact-embedding baseline, both admission guards together add four commands/scalar downloads per tiny step. Cutoff construction/scans/repeated validation add work. Native latency, peak memory and resource use are unmeasured; no speedup is claimed.

## Offline evidence

Evidence: `/workspace/exports/pocket-gpu-training-distilladmission-20261002`.
Gate script: `/workspace/tmp/pocket-distilladmission-gate.sh`.
Final window: **2026-10-02 11:33:46--11:35:15 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions |1,230 top-level +1,340 subtests|0|
| Selected tests, race detector |123 top-level +134 subtests|0|
| Affected CPU/mock regressions |824 top-level +1,339 subtests|1|
| Compiled-artifact follow-up |123 top-level +134 subtests|0|

Zero failures in final gates. Existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` is skipped with `requires separately reviewed fixed-corpus error-characterisation gate`: unrun, not passed, restrictions unchanged.

New tests check correct prediction/desired/logvar source edges, expanded graph/output counts, unchanged public vectors, source/result mutation isolation, missing outputs/dead malformed source, logvar range and standalone/expanded byte budgets. Flag count/shape/nonzero/nonfinite/sibling/caller alias rejects before backward. Contaminated host prediction and malformed vector results also reject despite a zero scalar flag. Owner/scope/inactive/invalid forward input failures allocate no tape.

Both new scope labels pass pinned all-gradient/full-step/update/EMA references, independent CPU objective endpoints and clipped updates, padding, repeated/absent/empty tokens and voices, checkpoint/resume, selected-output admission and all-tape failure retention. Exact new hash-kind/factory/opt-in tests pass. Numerical tolerances are unchanged; generated-signal hashes/byte identity are not acceptance gates. Exact replay comparisons test their structural contract only.

A later-row fault test injects a failing second distillation flag after earlier accumulation. It checks no further accumulation, zero optimiser/update-validation calls, zero returned metrics, unchanged trainer state and replay bytes, poisoned runner/backend, rejected checkpoint/dirty snapshot/Close and no retry. Workspace/sampling scratch may change while preparing the failed attempt; it is not a published completed boundary or a rollback guarantee.

Scope predicate and flag validator have100% scoped coverage; append helper96.0% and evaluator95.8%. Defensive errors after preceding complete source/shape admission remain partly unexercised. Vet, host build, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass. Offline CUDA12.8 sm80/PTX8.7 `--fmad=false`/ptxas and GLSL/spirv-val/conservative reflection pass; integration changes no kernels beyond the existing scalar primitive. Initial download-count expectations omitted the two added flags and were corrected; development logs are retained.

## Review limits and next work

An invariant judge emphasized later-row failure isolation and public/scope ownership rather than identifying a concrete new arithmetic bug. The targeted later-row publication/checkpoint tests above address the shared-state concern; previous gradient accumulation is intentionally not rolled back. Public calls, restore rejection and source ownership are tested. The judge did not inspect source or execute native code; independent source/native approval is still open.

Native cutoff/FTZ/rounding is unqualified, so redundant host admission remains. Device prediction/audio-hidden handoffs, removal of unnecessary downloads, EOS diagnostics, dense persistent gradient/optimiser work, full snapshots/live host copies and authorised native resources/performance/voice acceptance remain. All native/recovery/listening/publication holds are unchanged. The implementation goal remains active.
