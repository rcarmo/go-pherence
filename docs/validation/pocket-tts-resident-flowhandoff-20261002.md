## Transformer-to-flow resident handoff

Explicit `ptx-resident-flowhandoff-hybrid-v1` and `vulkan-resident-flowhandoff-hybrid-v1` scopes retain one read-only transformer/EOS forward producer across diagonal and distillation row consumers. Those private forward consumers bind the producer's audio-hidden root and select their condition row on device. Host audio-hidden/EOS overflow admission, prediction downloads, backward condition boundaries and full snapshots are retained.

CLI names omit `-v1`; options hash kind is `resident-flowhandoff-backend`. The new scope descends from distillation admission and compact embedding. Previous scopes, public forward builders, public minimal-LSD routing and replay identities are unchanged. Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`; staged inference remains eight paths / 469 insertions.

## Producer lifetime and consumers

Preparation checks handoff-driver capability before state allocation, in addition to inherited compact/lifetime/zero-sharing/state-check capabilities. The private forward path builds the existing complete source and three selected audio-hidden/EOSmean/finite-witness roots, then starts a zero-pruned `ResidentTapeLease`. The transformer executes once without accumulation. The producer's owned host output still passes the existing shape, finite, alias and absolute-dot EOS overflow guard.

For each active row, the backend records row/frame/hidden geometry and host-hidden slice provenance. `bindFlowHandoffCondition` validates complete source and bindings, locates the explicit `cond_embed.weight` forward input, follows only admitted reshape aliases back to its constant, and checks extent/value provenance plus parameter disjointness. It appends a full-hidden constant placeholder and vector mode-one row slice. Reads of the identified condition constants use the row result; old constants and IDs remain in full-source admission. The distinct producer root ID is checked by the lease, which remaps it and all live consumer edge IDs through pruning.

The lease preflights the complete expanded consumer source plus the live producer against one combined logical ceiling before consumer allocation. All bound parameters/edge buffers undergo kernel, extent, liveness and alias admission. Diagonal and distillation flags still precede backward, and redundant original host overflow admission remains. Parameter gradient roots, scales and order are unchanged; endpoint passes remain input-only and primary parameters accumulate once.

Other row backward/utility tapes use an authorised no-edge lease consumer. No-edge consumers also charge the live producer in their combined ceiling and retain the existing full-source/inventory/compact-index/accumulation preflight. The public state methods remain blocked while the lease is live. The lease closes after all rows, before final reduction/backbone and state update. Later graph work uses owned host boundaries and no longer references its native buffers.

All execution is caller-serialised and successful driver operations must be device-complete. No concurrent close/run contract is added. No implicit derivative crosses a handoff edge.

## Failure fencing

The step's named error path poisons a live producer before cleanup. The backend preserves its first terminal error and calls lease retention. A later-row guard failure keeps earlier resident gradients, blocks remaining accumulation, and publishes no metrics, update, replay, checkpoint or snapshot. Retry and ordinary close are refused.

Producer-close failure occurs after row work and before final reduction/backbone/update; it retains the producer and prevents publication. Cancellation after a row download also fences the producer and runner. Construction, execution and output failures are covered at each tape position; invalid row geometry, copied condition slices, missing inventory, unused nonfinite source and expanded-source ceilings reject before consumer allocation. Cross-scope restore into the preceding distillation-admission scope fails without trainer mutation.

## Numerical and accounting checks

Pinned independent PyTorch full-step loss, latent/parameter gradients, clipped multi-step updates and EMA pass with existing tolerances. Tests cover PTX/Vulkan scope routing, EOS/objective weights, padding, empty/repeated tokens, voice rows, replay/resume, public forward policy and late-row failure. The lifecycle mock binds actual producer arrays into the interpreter and verifies pointer provenance across row consumers, one producer close and no implicit forward accumulation. Older mocks retain their previous copied-parameter behaviour unless this explicit lifecycle hook is enabled.

| Tiny step | Distillation admission | Forward handoff |
|---|---:|---:|
| Tapes | 11 | 11 |
| Commands | 3,713 | 3,717 |
| Downloads | 46 | 46 |
| Downloaded elements | 121 | 121 |
| Planned storage bytes | 41,980 | 42,108 |

Four extra commands select rows in the two diagonal/two distillation forward tapes. Repeated forward-condition uploads are bypassed by native read-only binding. The host producer download remains necessary for current hidden/EOS guards and backward builders. Full-source accounting adds borrowed placeholders and charges the live producer; host metadata/constant construction still occurs. Native peak memory, complete-request transfer costs and latency have not been measured.

## Final offline gates

Window: **2026-10-02 12:59:47--13:01:17 UTC**. GPUs disabled, Go network access disabled, `GOMAXPROCS=2`, tests `-p=1 -parallel=1`. All final commands exited successfully.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 650 | 1,260 | 0 | 0 |
| Race detector | 65 | 126 | 0 | 0 |
| CPU/mock regression | 872 | 1,460 | 0 | 1 |
| Compiled-artifact follow-up | 65 | 126 | 0 | 0 |

`TestTrainingRandomMPFRComparisonDiagnostic` is the existing opt-in skip; it did not pass. Vet, host build, Linux ARM64/RISC-V builds, model-layout and diff checks passed. Offline CUDA/PTX and GLSL/SPIR-V compilation/validation passed; the slice adds no new kernel.

Scoped coverage: scope predicate, consumer wrapper, producer close and step helper 100%; condition rewrite 94.4%; forward evaluator 96.7%. Uncovered malformed graph-match/alias branches and native execution remain unqualified. Race tests exercise supported serialised ownership only.

Evidence: `/workspace/exports/pocket-gpu-training-flowhandoff-20261002`; script: `/workspace/tmp/pocket-flowhandoff-gate.sh`. `initial-pass/` preserves the preceding successful broader run. Early incorrect fixture fields, shape assumptions and replay whitelist omissions were corrected before final gates.

An abstract judge flagged tuple/provenance matching, replacement scope, close timing and first-error retention. Source checks enforce those constraints; the judge did not inspect source or native execution. No independent source approval was obtained.

## Remaining device work

Producer audio-hidden download, prediction downloads and backward host boundaries remain. Removing them needs scalar EOS/overflow contracts and retained forward/backward device result lifetimes. Dense persistent gradients/optimizer work, full snapshots and graph construction cost remain. Native FTZ, resources, throughput and new-voice listening require Rui/@default coordination; GPU probes, resets/recovery and model/audio runs stay on hold.
