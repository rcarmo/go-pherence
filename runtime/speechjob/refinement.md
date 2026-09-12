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

## HTTP boundary continuation

1. Problem: expose durable job operations without leaking source media or conflating request acceptance with completed background work.
2. User: one authenticated tenant behind an application-owned local/TLS boundary.
3. Success: bounded upload/status/run/cancel/delete and allowlisted JSON/VTT downloads.
4. Minimum: embeddable handler, fixed profiles, exact identity, synchronous execution, explicit shutdown.
5. Exclusions: listener/deployment, model loading, background queue, multi-tenant auth, browser UI, automatic retries.
6. Scope: new httpapi package, bounded Store.ListPage, tests/docs/Make target; separate worktree from adapter integration.
7. Platforms: core Go HTTP/store cross-build; existing lock/model platform restrictions unchanged.
8. Errors: generic codes; fresh stored snapshots after run errors; never echo private callback text/config/paths.
9. Persistence: store remains owner;201 only after durable upload;202 cancellation-request status is not a durable cancelled-state acknowledgement.
10. Patterns: existing explicit Store lifecycle and whole-recording serial executor; no token scheduler repurpose.
11. First change: auth before body reads, fixed-profile upload/status and artifact allowlist.
12. Avoid: arbitrary client config/model paths, CORS/cookies, raw PCM download, automatic retention cleanup.
13. Names: /v1/jobs and /v1/inventory; generated download filename and trusted artifact MIME types.
14. IO: bearer-token request, raw binary upload/profile/name; sanitised status and exact verified artifact bytes.
15. Compatibility: existing store/model APIs unchanged apart from additive ListPage; FFmpeg default unchanged.
16. Limits:512MiBmaximumupload,configured1..64requests+1cancel slot,one mutation,100statuspage,16MiBartifact;no model memory quota.
17. Trust: constant-time hashed token comparison, exact Host/Origin policy, no forwarded-header trust;TLS/server deadlines/cryptorandom token external.
18. Export: only transcript/vtt/speaker-transcript/speaker-vtt; explicit experimental content retained.
19. Proof: auth/body/routing/caps/ownership/profiledrift/error/integrity/disconnect/drain/reopen tests,realhttptesttransport and actual processkill.
20. Closure: scoped synthetic checks pass and local commit; production service/browser/resourcequeue/quality qualification remains open.

## HTTP CLI continuation

1. Problem: operate the tested HTTP job boundary without manual credential-bearing curl commands or unsafe partial downloads.
2. User: single tenant of an explicitly configured speech-job server.
3. Success: upload/status/run/cancel/delete and verified no-clobber artifact download.
4. Minimum: HTTP-only CLI with machine-readable JSON and signal-aware cancellation.
5. Exclusions: listener, model loading, queueing, browser UI, automatic retries/polling and private-network deployment.
6. Scope: new cmd/audio/speechjob plus Make/docs; no media/model/default changes.
7. Platforms: cross-compile Go client; local Linux filesystem/signal execution tested.
8. Errors: no token/raw private server errors; distinguish received/published state from failed local stdout.
9. Persistence: download tempfile→hash/size verification→file sync→no-clobber link→directory sync.
10. Pattern: existing HTTP exact profiles/IDs/artifact allowlist; no alternate inference runtime.
11. First change: URL/token/redirect policy, followed by exact-download tests and full httptest flow.
12. Avoid: tokens in flags, endpoint credentials, insecure TLS, redirects/proxy credential forwarding, blind POST retry.
13. Names: four transcript artifact names; local explicit destination never taken from response headers.
14. IO: environment token/endpoint, commands/files; bounded JSON stdout or verified 0600 transcript file.
15. Compatibility: HTTP wire/server/source-media/FFmpeg contracts unchanged.
16. Limits:4MiBJSON64KiBerror16MiBartifact512MiBupload;timeout30s default/8h maximum;no background workers.
17. Trust: HTTPS system verification, explicit literal-loopback HTTP only; stable trusted local files/directories; endpoint server trusted.
18. Export: only JSON/VTT allowed by HTTP API; experimental provenance preserved.
19. Proof: real fixture-handler flow, hash/length/MIME/redirection/race/cancel tests and actual SIGTERM child cleanup.
20. Closure: scoped tests/vet/cross-build/commit/evidence; no production service or trained-quality/performance completion.

## Explicit server/profile continuation

1. Problem: the HTTP handler/client need a real caller that binds trusted local model assets, store and listener lifetimes.
2. User: operator of one explicitly authorised local ASR profile.
3. Success: metadata checks before payload load/listen; owned store/model/HTTP shutdown; no implicit job execution.
4. Minimum: Linuxamd64 CPUWhisper+FFmpeg→rawjournal→conservativetranscript/VTT profile and command.
5. Exclusions: trainedqualification, Community/Vulkan serving, browser UI, queue, shared host resource controller and deployment.
6. Scope: cmd/audio/speechjobserve with Make/docs/tests; existing inference math unchanged.
7. Platforms: Linuxamd64 model execution/check; other builds fail explicit profile support, help available.
8. Errors: fail startup before listener on invalid metadata/config/token/TLS; close store on load failure; do not leak paths/tokens.
9. Persistence: existing private store lock before weights; recover interrupted jobs but never run automatically.
10. Pattern: checked model/config/generation loader and existing four-stage job pipeline; preserve FFmpeg default.
11. First change: strict config/pinned asset/metadata caps, then owned bounded HTTP server.
12. Avoid: downloads, implicit language/runtime modes, post-init GPU disable claims, arbitrary client model configuration.
13. Names: explicit local assets and profile ID; template has invalid hashes and allow_execution false.
14. IO:64KiBconfig,hashpinnedmodel/tokenizer/generation/executables;oneHTTP profile;metadata-only --check.
15. Compatibility: existing services untouched;NVIDIA disabled before exec,GOMAXPROCSmatchesconfig;nondefaultWhisperflags rejected.
16. Limits: store/file/header/tokenizer/widenedweights/network caps and separate header/socket/handler timers;not RSS orpreemptivekernelcontrol.
17. Trust: adminimmutable paths/hashreopen boundary,TLSvalidity+hostcoverage,envtoken,exactHost/Origin;sharedcompute external.
18. Export: existing allowlisted transcript/VTT APIs;not trainedqualified output.
19. Proof: generatedtoy safetensors/vocabulary+realFFmpeg,inprocessTLS/cap/deadline/drain,realSIGTERMchild lock/listenerrelease.
20. Closure: scoped tests/vet/build/commit/evidence;no productionlistener/service/model/performance acceptance claim.
