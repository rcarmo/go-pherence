## Transformer-to-flow handoff integration plan

The transformer produces one `Frames*Hidden` audio-hidden buffer per training step. `training_step.go` then runs multiple diagonal/distillation row consumers before the final backbone VJP. `runResidentTape` currently downloads its requested boundary and closes the producer tape. The qualified [two-tape primitive][pair] cannot directly serve all these consumers without a longer-lived coordinator lease.

The [coordinator lease primitive][lease] is implemented and qualified offline, including multiple consumers, compact accumulation, remapped roots, failure retention and state guards. The [opt-in model/CLI integration][integration] now retains the producer across row forward consumers and authorised no-edge backward tapes. The original host hidden/EOS guards and all existing downloads remain. Older scopes/replay identities and execution holds are unchanged; new scopes have distinct identity.

## Coordinator lease

Use the explicit `ResidentState` producer lease with owned tape, retained output roots, complete source/binding admission and a logical storage charge. The producer binds current resident parameters and runs once without accumulation. The state must prevent update, snapshot, ordinary close and unrelated public accumulation while a lease is live, while allowing only its own explicitly authorised consumer execution and gradient accumulation. A lease cannot simply leave `s.borrowed=true`: existing `s.admit` rejects every later tape call.

The primitive separates the long-lived lease flag from invocation-local tape borrowing. Keep the existing public guards; use the lease-authorised consumer route that validates lease identity, state identity and liveness before using it. Never expose a free-standing current-parameter map. On successful completion, close all consumers before the producer and clear the lease only after cleanup succeeds. On execution, accumulation, validation-after-execution or close errors, poison the owning training boundary and retain all potentially referenced tapes. Previously accumulated rows are not rolled back.

The combined logical ceiling must include the pinned producer and each simultaneously live consumer, including expanded admission tensors. Complete source/inventory admission precedes pruning. Remap declared roots and constant-target edge IDs through pruning while preserving compact embedding indices and gradient roots. Existing lifetime/positive-zero allocation paths protect borrowed edge placeholders.

## Row consumers

Keep the producer's audio-hidden root pinned. For each row, create a read-only row slice inside the consumer graph with the existing vector mode-one slice utility, without adding a backend sub-buffer API. The consumer borrows the complete producer root as an explicit constant, then selects the row. Admit row range, hidden extent, output declaration, alias provenance and all external parameter bindings before allocation. This avoids pointer-offset/lifetime ambiguity, although the conservative logical ceiling charges the full borrowed extent.

The first integration uses diagonal/distillation forward consumers only. Their backward calls can continue using the host row boundary initially, preserving the public builders and result ownership. Endpoint backward remains input-only; primary parameter gradients accumulate exactly once with the original scale. Do not add gradients automatically across the borrowed edge. Full forward/backward handoff needs a separate reviewed extension.

Keep the audio-hidden host download initially: `trainingStepTapes` constructs each row's `Condition` from that array, and `decodeForwardResults` performs the EOS absolute-dot overflow admission. The first opt-in scope can avoid repeated row-condition upload while retaining these guards; it must not advertise removal of the producer download. A later scalar EOS/admission contract must replace both needs before that download can disappear.

## Opt-in routing and tests

Use a distinct scope/replay/hash identity descended from distillation admission and compact embedding. Add capability checks before state allocation, CLI factory mappings, cross-scope restore rejection and owner checks. All existing public builders, legacy scopes and replay identities stay unchanged.

Test transformer execution once with multiple row consumers, borrowed-root identity with no condition upload, scalar flags and redundant host guards before backward, masks/empty and repeated tokens, full source/late combined ceilings, pinned full-step loss/all-gradient/update/EMA parity, endpoint stop-gradient and exact primary accumulation count. Inject a later-row failure and verify no metrics, update, replay, checkpoint, snapshot, retry or ordinary close despite earlier gradients. Compare logical commands/downloads/uploads/bytes explicitly; native latency and memory need coordinated hardware runs.

Native GPU execution, FTZ checks, model/new-voice quality, listening and publication require Rui/@default coordination. No recovery, reset or probe is authorised by this plan.

[pair]: pocket-tts-resident-tapehandoff-20261002.md
[lease]: pocket-tts-resident-tapelease-20261002.md
[integration]: pocket-tts-resident-flowhandoff-20261002.md
