# Speech-job foundation scope

This implements the persistence slice of the existing speech plan, not a new service or inference backend. Requirements are taken from `docs/speech-integration.md` and the approved source plan; no model/service defaults change.

1. Problem: uploads and completed stages must survive later stage failure and process death.
2. User: recording jobs submitted through a future Go CLI/HTTP workflow.
3. Success: acknowledged original+manifest are durable; verified checkpoints resume without re-executing successful work.
4. Minimum slice: bounded local store, serial stage executor, cancellation, inventory/download/retry/delete APIs.
5. Exclusions: model loading, worker creation, automatic retries, CLI/HTTP endpoints, browser UI, distributed storage.
6. Files: `runtime/speechjob`, its tests, and scoped Make/documentation targets.
7. Platforms: cooperating-process flock on Unix; tested Linux; unsupported platforms fail explicitly.
8. Errors: retain original/checkpoints, publish failed/cancelled status; callbacks must drain their own activity.
9. Persistence: fsync payload, no-clobber content publication, fsync directory; fsync temporary manifest, atomic rename, fsync directory.
10. Existing patterns: Go model/media interfaces remain independent; token prefill/decode scheduler is not repurposed for whole-file jobs.
11. First change: storage/identity tests before real inference adapters.
12. Avoid: generic tensor store, distributed queue, model fallback, service activation, inference-speed claims.
13. Naming: random job IDs and generated hash-keyed blob names; upload name is inert metadata.
14. Inputs/outputs: caller-owned Readers, UTF-8 JSON configuration and versioned stages; owned manifest snapshots and verified read-only artifacts.
15. Compatibility: existing CLI and model APIs untouched; FFmpeg default unchanged.
16. Limits: one running job per store, explicit job/file/total-byte caps; no inference-memory budget supplied by this package.
17. Security: private caller-owned directory, rooted IO, symlink inventory rejection, advisory cooperating-process lock. No malicious same-UID writer protection.
18. Export: verified completed stage payloads remain readable after later failure; no user-facing raw-upload download endpoint.
19. Proof: corruption/key mismatch tests, fault injection, real child SIGKILL at publication boundaries, concurrency, cancellation/retry, quota and interrupted-delete recovery.
20. Completion: the persistence foundation and explicit FFmpeg decode/JSON/WebVTT slice pass scoped tests. The full plan requires model adapters, scheduling, CLI/HTTP/UI, long-job and deployment qualification separately.

## Media/transcript continuation

The next tested slice uses the temporary FFmpeg backend, preserving original input and publishing fixed-header canonical PCM. Configured executable hashes, private scratch, pre-admission file-byte accounting, explicit crash-scratch retention and cancellation drains extend the existing lifecycle contract. Synthetic WAV and AAC/M4A tests use no model or private recording.

Transcript schema 1 stores already reconciled sample-indexed cues. JSON requires exact complete keys; WebVTT escapes cue text and generates local speaker labels. These APIs do not align words, reconcile overlapping windows, identify speakers or map source PTS. Decoder CPU/RSS/OS disk quotas and user-facing interfaces require separate work.

## Whisper journal continuation

1. Problem: a failed long ASR stage must preserve completed windows without the store's 64-stage cap.
2. User: existing planned recording-job workflow.
3. Success: verified window prefixes skip PCM reads/inference on retry.
4. Minimum: host Go Whisper adapter, per-window payload/ack journal and bounded raw JSONL result.
5. Exclusions: trained qualification, Vulkan retained-work drain, final text reconciliation, Community-1 and service/UI.
6. Scope: speechjob adapter/tests; checked Whisper resume entry point and host-eligibility validation.
7. Platform: experimental adapter is Linux/amd64 only; existing model-independent store retains scoped cross-build coverage. Wider Whisper ARM64/Windows dependencies fail to build. Shared model/backend ownership remains external.
8. Errors: failed window is not acknowledged; prior acknowledgements survive; callback errors fail closed.
9. Persistence: fsync/no-clobber payload, then fsync/no-clobber size/hash acknowledgement and directory sync.
10. Pattern: existing deterministic stage publication and explicit retry.
11. First change: suffix-only window orchestration, without changing the default start-at-zero API.
12. Avoid: generic tensor store, broad scheduler rewrite, automatic fallback or model runs.
13. Names: dependency-keyed window payload/ack files; `asr-windows` final stage.
14. IO: verified decode PCM, immutable model/tokenizer, explicit options; raw canonical-second window JSONL.
15. Compatibility: existing APIs/defaults unchanged; the new resume method is explicit.
16. Limits: four hours/10,000 windows; 1 MiB/window, 64 MiB/result; conservative file reservation, no CPU/RSS budget.
17. Trust: caller-attested loaded weight/runtime hashes; model/tokenizer/backend flags immutable; host-only validation rejects GPU buffers/features.
18. Export: final raw-window checkpoint only after complete ASR; it is not final transcript JSON/VTT.
19. Proof: toy actual frontend/encoder/decoder, >100 windows, corruption/gaps/orphans/quota/cancel/reopen and SIGKILL payload/ack boundaries.
20. Closure: scoped tests/evidence pass. Full workflow and trained-quality/performance acceptance remain open.

## Community-1 stage continuation

1. Problem: attach existing experimental diarization to durable recording jobs without losing completed ASR on failure.
2. User: existing planned Go recording workflow.
3. Success: complete raw diarization checkpoints survive restart and are skipped on retry.
4. Minimum: caller-loaded CPU model adapter and compact validated raw-turn document.
5. Exclusions: trained quality, neural-window journals, final text alignment, long-file scaling, service/UI/GPU work.
6. Scope: speechjob adapter/tests and documentation; model numerical code unchanged.
7. Platform: Linux/amd64 adapter; core store retains scoped cross-build coverage.
8. Errors: preserve prior ASR/transcript/VTT; repeat failed diarization from start; no fallback.
9. Persistence: existing whole-stage atomic fsync/hash/no-clobber checkpoint contract.
10. Pattern: existing explicit experimental Go model wrapper and serial job executor.
11. First change: bounded policy/geometry/result validation with injected tests, then actual reduced synthetic model wiring.
12. Avoid: changing tolerances/ties, interpreting raw turns as word alignment, promoting defaults.
13. Names: `diarization` stage; schema-1 `DiarizationDocument` canonical JSON.
14. IO: verified decode PCM, immutable caller model, complete policy/modes; raw full/exclusive turns and diagnostics.
15. Compatibility: FFmpeg and neural defaults unchanged; requires explicit AllowExperimental.
16. Limits: model128windows/fourhour outer bound,16MiBoutput;137s maximum with10swindow/1sstep;file admission only.
17. Trust: caller-attested checkpoint/filter/runtime hashes, immutable private root/payloads/model; no hostile same-UID writer guarantee.
18. Export: validated complete raw diarization checkpoint; prior text remains downloadable after failure.
19. Proof: reduced synthetic silence/single/clustered model paths, direct equality, repeat determinism, bounds/cancel/retry and3SIGKILL boundaries.
20. Closure: scoped checks/evidence pass; full workflow/neural/corpus/performance acceptance stays open.

## Conservative transcript reconciliation continuation

1. Problem: convert complete ASR windows to usable text without silently deleting conflicting overlap or inventing alignment.
2. User: planned recording workflow with text available despite diarization failure.
3. Success: exact duplicates merge, disjoint text survives, conflicts fail with raw output retained.
4. Minimum: explicit exact-overlap transcript stage plus separate conservative experimental speaker output.
5. Exclusions: fuzzy matching, word alignment/interpolation, corpus-quality or source-PTS claims.
6. Scope: speechjob reconciliation/speaker stages/tests/docs; no neural algorithm changes.
7. Platform: Linuxamd64 alongside model adapters; corecross unaffected.
8. Errors: no final checkpoint on conflict/invalid stream; preceding outputs stay downloadable.
9. Persistence: existing atomic whole-stage store; plain text before diarization, speaker output separate.
10. Pattern: verified prefix/dependency keys and versioned deterministic policies.
11. First change: canonical complete-window parser then exact overlap checks.
12. Avoid: dropping segments by window ownership or selecting exclusive speaker turns to hide overlaps.
13. Names: transcript/vtt and speaker-transcript/speaker-vtt; provenance/experimental status in speaker document.
14. IO: complete raw ASR+decoded extent, declared matching ASRversion/language/geometry; optional full diarization turns.
15. Compatibility: existing schemas/defaults unchanged; new stages explicit.
16. Limits:64MiBstream1MiBrecord100000rawcues16MiBoutput;bounded sort and64speaker binary searches/cue.
17. Trust: configured language matches ASRversion; immutable checkpoint bytes; experimental input accepted explicitly only.
18. Export: plain JSON/VTT first; speaker JSON/VTT with warning and source keys separately.
19. Proof: exactduplicate/conflict/bounds/geometry/samplegrid/provenance/speakercoverage tests and3newkillpoints.
20. Closure: scoped deterministic tests pass; full word/overlap/model/performance qualification stays open.
