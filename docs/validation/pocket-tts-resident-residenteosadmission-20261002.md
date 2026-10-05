## Opt-in producer EOS admission

`ptx-resident-eosadmission-hybrid-v1` and `vulkan-resident-eosadmission-hybrid-v1` append a conservative EOS projection-admission flag to the retained transformer/EOS producer. The training step checks that scalar before any row gradient accumulation. It keeps the audio-hidden download and original host F64 absolute-dot guard.

CLI names omit `-v1`; hash kind is `resident-eosadmission-backend`. The new scopes inherit forward handoff, distillation/diagonal admission and compact embedding. Older scopes, public forward builders and replay identities are unchanged. Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`; staged inference remains eight paths / 469 insertions. Native GPU/model/audio/listening/publication holds are unchanged.

## Complete admission and provenance

`appendForwardEOSAdmission` validates the complete producer graph and resident binding inventory before changing it. It admits frame/hidden geometry, selected hidden extent, distinct EOS weight/bias binding IDs and extents. It locates the exact affine command using those bound parameters, follows reshape aliases, and requires its hidden slice to share the selected audio-hidden source and offset. Similar tensor shapes alone cannot establish provenance.

The helper copies tensor/command/output lists, appends an EOS-weight flattening reshape and an `eos-admission` command, and declares its scalar output. Full expanded source and binding admission precedes lease construction/pruning. The original graph IDs, commands, output arithmetic, parameter-gradient roots and weight/scaling policy are unchanged. Both execution branches use the selected output list captured before appending the flag; old three-root selection is never recomputed from shifted output positions.

The zero-pruned producer lease retains four roots: audio hidden, EOS mean, finite witness and admission flag. Native binding/kernel preflight occurs before allocation/execution. The flag checks all rows, including padding; it has no gradient. Parameter accumulation is disabled for the producer.

The returned flag must have exactly one finite zero value, be disjoint from the other outputs, and not borrow normalized/voice input arrays. After checking it, the existing three-root decoder still validates shape/finite/ownership and the original host absolute-dot bound. A forged zero flag therefore cannot bypass host overflow rejection. The same producer survives later read-only row consumers and closes before final reduction/backbone/update.

## Conservative scope policy

The [integer primitive][primitive] computes an upward absolute-dot bound in `2^95` units under the original limit and reserves one unit for host accumulation rounding. The bound can reject host-safe near-limit values; this conservative policy is explicit in the new scope and CLI notice. It does not replace or relax the host guard.

A producer fixture sets zero EOS weights and an original-host-safe bias just below the limit. The appended flag rejects it as expected, including the padded-row policy. Ordinary pinned fixtures pass unchanged. The primitive's independent F64 product/whole-row checks and native execution limits are recorded separately.

## Failure and replay fencing

Flag rejection, malformed extent, NaN, output alias and borrowed flag tests stop the runner before any accumulation or update. They retain the producer lease, return zero metrics and preserve trainer/replay state. Checkpoint, snapshot, retry and ordinary close fail afterward. The zero-flag/overflowed-hidden test confirms the redundant host guard has the same ordering.

Preparation checks inherited handoff capability before resident allocation. The exact new scope inherits handoff/compact/diagonal/distillation policies; a focused test fixes that relationship and verifies first-error retention through `b.fail` and `lease.RetainFailure`. The lease remains terminal and cannot close or publish state.

PTX/Vulkan full-step, all-gradient/update/EMA, objective-weight, clipping, padding, voice/token, public-route, per-tape construction/output failure and replay/resume tests pass. Cross-scope restore is rejected without trainer mutation. A lifecycle test verifies one admission command before gradients and one producer close after row accumulation.

## Tiny-step accounting

| Tiny step | Forward handoff | EOS admission |
|---|---:|---:|
| Tapes | 11 | 11 |
| Commands | 3,717 | 3,719 |
| Downloads | 46 | 47 |
| Downloaded elements | 121 | 122 |
| Executed logical bytes | 134,236 | 134,256 |
| Planned storage bytes | 42,108 | 42,108 |

The two extra commands flatten the borrowed EOS weight and check the flag. Weight reshape aliases existing storage; lifetime planning reuses a scalar slot, so the measured planned storage is unchanged. This adds diagnostic work and one scalar transfer. Native latency, allocator overhead, peak memory and complete-request performance have not been measured. Hidden, prediction/backward downloads and full snapshots remain.

## Final offline gates

Window: **2026-10-02 13:36:43--13:38:33 UTC**. GPUs and Go network access disabled, `GOMAXPROCS=2`, tests `-p=1 -parallel=1`. All final commands exited successfully.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 610 | 920 | 0 | 0 |
| Race detector | 61 | 92 | 0 | 0 |
| CPU/mock regression | 904 | 1,508 | 0 | 1 |
| Compiled-artifact follow-up | 61 | 92 | 0 | 0 |

`TestTrainingRandomMPFRComparisonDiagnostic` is the existing opt-in skip; it did not pass. Vet, host build, Linux ARM64/RISC-V builds, model-layout and diff checks passed. Offline CUDA/PTX and EOS-admission GLSL/SPIR-V compilation/validation passed. Scope and flag validation reached 100% scoped coverage; appender 96.9%, private forward evaluator 94.9%. Malformed unmatched paths and native execution are unqualified. Race gates cover supported caller-serialised use only.

Evidence: `/workspace/exports/pocket-gpu-training-residenteosadmission-20261002`; script: `/workspace/tmp/pocket-residenteosadmission-gate.sh`. `initial-pass/` preserves the preceding passing run. Early test accounting omitted the additional diagnostic and was corrected. EOS slice sources required reshape-family normalisation before provenance matching.

A focused source judge raised a conditional non-handoff output-selection issue and did not inspect the failure helper. EOS-admission already inherits handoff, making that conditional path unreachable; both branches now use the captured output list anyway. Explicit inheritance/first-error/lease-retention tests passed. The follow-up source judge timed out after 45 seconds with no verdict. No independent whole-source or native approval was obtained.

## Remaining transfers

Prediction/backward result lifetimes, hidden download removal, full snapshots and dense persistent gradient/optimizer work remain. Keep original guards until independent source and native numerical/FTZ/resource qualification permits further changes. New-voice training and listening require Rui/@default coordination; no probe/reset/recovery/model/audio execution occurred.

[primitive]: pocket-tts-resident-eosadmission-20261002.md
