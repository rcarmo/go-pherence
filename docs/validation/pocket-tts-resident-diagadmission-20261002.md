## Resident diagonal prediction admission

Two opt-in scopes now append scalar prediction admission to the private read-only diagonal forward tape:

- `ptx-resident-diagadmission-hybrid-v1`
- `vulkan-resident-diagadmission-hybrid-v1`

They inherit compact embedding, selected private results and the existing device optimiser. Predictions still download, and the original host admission remains as a redundant check. This is diagonal-only integration; distillation admission and device handoffs remain open. No transfer reduction or native performance improvement is claimed.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths/469 insertions. Execution used offline CPU/mock tests with NVIDIA/Vulkan disabled; shader/PTX compilation was offline only. No native/model/audio/recovery/publication work occurred.

## Ordering and admission

The step knows diagonal log variance before forward execution. A private optional hook carries it into the new scope, without changing the public diagonal-forward interface or its result fields. Older scopes retain their original call path.

`appendDiagonalAdmission` validates complete source tape/bindings before copying metadata and adding a cutoff constant plus scalar mode-two LSD command. That command reads the admitted prediction and desired tensor IDs. It uses the maximal-F32 cutoff derived from the original F64 square/seed-overflow comparisons, described in [scalar primitive validation](pocket-tts-resident-lsdadmission-20261002.md). The original standalone loss byte ceiling and expanded forward graph/binding ceilings are both retained before pruning/allocation.

The tape runs with accumulation disabled. It downloads the existing input/desired/prediction vectors and one scalar flag. Shape, finite-zero flag, sibling/caller alias checks and vector ownership/finiteness checks precede return. The original `admitLSDLossTraining` then rechecks downloaded values. Only a successful return reaches the diagonal backward call. The host check is deliberately retained pending native rounding/FTZ qualification and a later device handoff.

Original tensor IDs, command order, outputs, source values and bindings remain unchanged; only owned appended metadata is added. Separate builds own their flag inputs/cutoff constants/output headers. Unchanged nested metadata stays read-only borrowed. Parameter scales, endpoint stop-gradient, accumulation roots and public FlowLM/diagonal APIs remain unchanged.

This is a per-row gate, not whole-step gradient atomicity. Earlier successful rows may already have resident gradients when a later row fails. Such failure remains terminal: no trainer update/replay/checkpoint publication or retry is permitted, and uncertain resource owners remain retained. It does not release any recovery hold.

## Scope and transfer costs

CLI names are `ptx-resident-diagadmission-hybrid` and `vulkan-resident-diagadmission-hybrid`. Explicit experimental-device opt-in, runtime pin and logical byte ceilings remain required. Factory coverage includes both scopes/component owners. Hash kind is `resident-diagadmission-backend`; same-scope resume passes and restore into earlier compact/pruned scopes is rejected without mutation. Older scope hashes remain unchanged.

Tiny complete-row CPU/mock accounting versus compact-embedding baseline:

| Measure | Baseline | Diagonal admission |
| --- | ---: | ---: |
| Tapes |11|11|
| Commands |3,709|3,711|
| Downloads |42|44|
| Downloaded elements |117|119|
| Executed logical tensor bytes |134,028|134,044|
| Planned storage bytes |41,948|41,964|

There are two valid diagonal rows, each adding one command, one scalar download and eight logical bytes (cutoff plus flag). Accumulation counts are unchanged. Guard scanning, host cutoff derivation and expanded validation add work; predictions still transfer. Native resource/latency and complete-request speed are unmeasured. Persistent dense gradients/optimiser work and snapshots remain.

## Tests and evidence

Evidence: `/workspace/exports/pocket-gpu-training-diagadmission-20261002`.
Gate script: `/workspace/tmp/pocket-diagadmission-gate.sh`.
Final full gate: **2026-10-02 11:23:47--11:25:17 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions |980 top-level +1,030 subtests|0|
| Selected tests, race detector |98 top-level +103 subtests|0|
| Affected CPU/mock regressions |800 top-level +1,308 subtests|1|
| Compiled-artifact follow-up |98 top-level +103 subtests|0|

Zero failures in final gates. Existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`: unrun, not passed, with no restriction changed.

Tests verify appended command source IDs/logvar/cutoff, original vectors and output ordering, source/result mutation isolation, invalid/dead sources, missing boundary outputs, logvar range, standalone/expanded budgets, inactive/foreign owner and scope rejection. Invalid flags (shape/nonzero/nonfinite/alias) reject before diagonal backward. A zero flag plus contaminated prediction is still rejected by the host guard. Borrowed caller scalar storage is rejected. Failure leaves trainer/run state unpublished and blocks retry/Close.

Both scopes pass pinned full-step/all-gradient/update/EMA tests, independent CPU objective endpoints, clipped updates, padding, seeded/repeated/absent/empty-token conditioner gradients, checkpoint/resume, output admission and all-tape failure injection. Existing public forward calls still return the same numerical values without the private guard. Tolerances are unchanged; generated-signal hashes/byte identity are not acceptance gates. Exact replay comparisons cover the replay contract only.

Scope predicate and flag decoder have100% scoped coverage; append helper96.0% and evaluator92.6%. Defensive errors after preceding full source/shape admission remain partly unexercised; coverage is not approval. The initial accounting expectation omitted the new scalar transfers and the direct-call fixture used mismatched model dimensions; tests were corrected and development logs retained.

Vet, host build, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass. Offline CUDA12.8 sm80/PTX8.7 `--fmad=false`/ptxas, GLSL/spirv-val and conservative reflection pass. This integration adds no kernel beyond the earlier scalar primitive.

## Review and remaining work

An invariant judge found no concrete break in the stated design, emphasizing full-source validation, read-only ownership, guard-before-backward, scope rejection and precision assumptions. Bounds/source/flag/ordering/failure/replay tests address those conditions. Native FTZ/rounding is not assumed proven; redundant host admission stays in place. The judge did not inspect source or execute native code, so independent source/native approval remains open.

Next qualify distillation scalar admission and device prediction/audio-hidden handoffs without weakening guards. Dense persistent gradients/optimiser work, EOS diagnostics, full snapshots/live host copies and authorised native resources/performance/voice acceptance remain. These are resident hybrids, not full-device training. All native/recovery/listening/publication holds remain; the overall implementation goal is active.
