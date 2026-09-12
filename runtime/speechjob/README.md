# Durable speech-job foundation

`speechjob` stores acknowledged uploads and sequential stage checkpoints with explicit retry, cancellation and retention. It also provides an explicitly configured FFmpeg decode stage, host Go Whisper raw-window and experimental Community-1 diarization stages, plus transcript JSON/WebVTT serializers. It does not load models or start a service. The optional [durable queue](queue.md) requires explicit construction and worker startup. An [experimental resident Vulkan Whisper stage](whisper-vulkan.md) adds explicit native drain/quarantine ownership without changing the CPU default. An [experimental Community-1 CPU owner](community1-owner.md) serialises synchronous diarization and clears exclusively transferred Go model references after drain. The separate [HTTP handler](httpapi/README.md) provides authenticated synchronous or explicitly queued job operations and transcript-only downloads when embedded by an application. The existing inference scheduler remains unchanged: its token-prefill/decode contract does not model whole recording stages.

This is the persistence slice of the speech plan. [Refinement scope](refinement.md) records its requirements and exclusions.

## Store and execution contract

- `Open` creates or opens a private store beneath an existing caller-owned parent. It fsyncs the parent directory and takes an advisory process lock. Supported lock implementations cover Unix platforms; other platforms fail explicitly. Linux is the tested platform.
- `Create` publishes the original upload before returning a queued manifest. It writes a generated temporary filename, fsyncs it, publishes with a no-clobber hard link, then writes/fsyncs/atomically replaces the manifest and syncs the directory. Upload names are inert metadata.
- `Run` admits one job at a time per store and invokes stages sequentially in the caller's goroutine. It verifies original bytes and every retained checkpoint before reuse. It does not spawn a worker, perform automatic retry or use model-specific logic.
- `Cancel` signals only the admitted job. The stage callback must honour context cancellation and drain its own GPU/worker activity before returning. Arbitrary blocking readers and callbacks cannot be forcibly interrupted by this package.
- `OpenCheckpoint` verifies length/hash before exposing a read-only payload, even after a later stage has failed. A future HTTP download allowlist must decide which stage names are user-downloadable; this API does not itself classify transcript versus PCM artifacts.
- `Inventory` and `Usage` include incomplete uploads and orphaned files. They count against explicit limits. `Delete` is the explicit retention-cleanup entry point; it first renames the job to a durable deletion marker, then removes it. A crash can leave that marker visible for an explicit retry. There is no automatic retention deletion on open.

No source path, model weight or private recording is embedded in the package. All readers and returned manifest snapshots have explicit ownership. The caller closes returned readers and the store. `Close` refuses a store with active execution.

## Identity and resume

The configuration is opaque, bounded UTF-8 JSON retained byte-for-byte. Callers must include the complete semantic identity: audio/backend/model revisions, precision, frontend and chunk geometry, language/task, decoding options and any speaker/tie policy. This package cannot detect an omitted option. Whitespace changes deliberately change configuration identity.

Every stage has a name and SHA256 version contract. A checkpoint key includes the original upload hash, configuration hash, stage name/version and all preceding checkpoints in order. Existing checkpoints must be an exact prefix of the supplied execution plan. Changing the configuration, ordering or version rejects resume; no old result is silently relabelled.

A stage must produce byte-deterministic output for its key. If a process dies after payload publication but before manifest publication, retry can reuse only identical payload bytes. Different output is rejected without overwriting the orphan. Nondeterministic model execution/export therefore requires a distinct configuration/job or a separately designed attempt policy; it is not hidden behind a claimed deterministic resume.

A completed job is a verified no-op only for its original full stage list. Failed and cancelled jobs require an explicit `Run` to retry. Opening a store marks interrupted `running` jobs failed, preserving original media and completed checkpoints; it never resumes inference itself.

## Publication failures

If a persistence operation fails after rename, the durable state can be old or new. `ErrPersistence` distinguishes manifest-publication errors. A nonnil error may accompany an in-memory proposed manifest; do not treat it as durable success. Inspect `Get`, `Inventory`, or reopen the store before making a retry/deletion decision.

Failed/unacknowledged uploads are retained and exposed by inventory. Corrupt manifests/payloads fail closed; opening a store with a corrupt acknowledged manifest requires explicit operator repair outside this API. Startup does not discard corrupt data to make progress.

Progress callbacks observe committed snapshots and must not block indefinitely. Their panics are contained and ignored. Stage callback panics become failed jobs. Context cancellation still stops further work. Successful final publication is not undone by a reporting failure.

## Storage and platform limits

The required filesystem supports same-directory atomic rename, hard links, file fsync and directory fsync. The guarantees cover cooperating processes using this API on a caller-owned private local directory. They do not cover malicious same-UID writers, distributed filesystems, external mutation after verification, filesystem corruption or hardware that ignores flushes. Rooted file operations confine access, and symlink entries found in store inventory are rejected.

Limits bound job count, upload size, stage payload size and total file bytes. Total accounting includes old/orphaned files and manifests, with conservative headroom for publication. The package does not impose model memory, subprocess RSS, CPU or GPU budgets. Mutating calls reject a busy store rather than oversubscribe it. Large recursive inventory/deletion calls are synchronous.

## Usage shape

```go
store, err := speechjob.Open(privateDirectory, speechjob.Limits{
    MaxJobs: 32,
    MaxUploadBytes: 256 << 20,
    MaxArtifactBytes: 512 << 20,
    MaxBytes: 4 << 30,
})
// Check err; defer store.Close().
job, err := store.Create(ctx, "recording.m4a", versionedConfigJSON, upload)
// Queue acknowledgement is valid only when err == nil.
result, err := store.Run(ctx, job.ID, versionedConfigJSON, stages, progress)
// Stage functions, versions, model admission and cancellation drains are caller-owned.
```

This is an API sketch, not a running CLI or server. Trained model-stage qualification, general word/overlap alignment, a deployed CLI/HTTP service, durable queue admission and UI progress remain unimplemented. The separate HTTP package has a tested transcript-download allowlist; it opens no listener itself. The explicit exact-overlap reconciliation policy below handles only an unambiguous subset.

## FFmpeg decode checkpoint

`NewFFmpegDecodeStage` creates a `decode` stage from explicit absolute binary paths, SHA256 executable identities, input extension and positive input/output/duration caps. No executable runs during construction. The paths and their parent directories must be administrator-controlled and immutable throughout the job. Hash verification detects existing changes; execution by path has a concurrent-replacement race. Shared libraries and OS identity must also appear in the job configuration.

The stage materialises the verified upload inside a private job-owned `.work-decode-*` directory. It calls the existing media adapter, reopens the result through `OpenCanonicalPCM`, checks format/count/size against the adapter result, then writes a fixed 44-byte RIFF header and losslessly re-encoded s16 samples to the store's checkpoint writer. Output metadata and temporary filenames cannot affect checkpoint bytes. Actual decoded samples define the timeline; the stage does not trim AAC padding or map source-container edits/PTS.

Admission counts all retained files and reserves an input copy, two maximum-sized outputs (decoder scratch and published checkpoint), and manifest headroom. The output cap must also fit the store's artifact limit. The adapter's `-fs` limit and size monitor can overshoot during subprocess buffering/scheduling; this reservation is not a hard OS disk/RSS quota. Callers need external worker resource admission. The private store root must not be moved or renamed while open.

Normal success, failure, panic and drained cancellation remove the current scratch directory. Process death can retain scratch, which inventory includes in job bytes. Reopen and retry preserve old scratch; it can prevent another decode admission. Explicit `Delete` removes the job and its retained scratch. These files are temporary work, not acknowledged checkpoints, and are not separately fsynced for power-loss persistence.

## Transcript JSON and WebVTT

`Transcript` schema 1 uses integer mono-16-kHz sample timestamps with a four-hour maximum. Cues have positive spans inside the recording and nondecreasing starts; overlaps are allowed. `Speaker=-1` is unlabelled, and 0–63 are caller-assigned local IDs. Language is a 2–32-byte lowercase ASCII/hyphen identifier. Limits are 100,000 cues, 65,536 UTF-8 bytes per cue, and 16 MiB per serialised document.

`WriteTranscriptJSON` validates before writing and canonicalises empty cues to `[]`. `ReadTranscriptJSON` rejects unknown, missing, duplicate and case-aliased keys, null values, invalid input UTF-8 and excessive nesting. Serialisers cannot infer an omitted language, speaker, word alignment or overlap policy. The caller supplies reconciled final cues; raw overlapping Whisper-window callbacks are not final transcripts.

`WriteWebVTT` floors starts and ceils ends to milliseconds, adding less than 1 ms at each boundary and preserving sub-millisecond spans. It escapes ampersands and angle brackets with WebVTT-supported named references, preserves literal quotes, replaces newlines/tabs with spaces, and generates speaker voice tags. Text cannot insert cue separators or markup. `NewVTTStage` reads the verified `transcript` checkpoint and emits `vtt`; its version includes the escaping and strict-input contract. Output errors or cancellation can leave a partial caller writer; `Store.Run` never publishes it as a completed checkpoint.

## Go Whisper window journal

On Linux/amd64, `NewWhisperWindowStage` binds a caller-owned, immutable Go `Whisper` model and tokenizer to `asr-windows`. The adapter is build-tagged to this platform because the wider Whisper/backend dependency graph does not cross-build for ARM64/Windows. The model-independent store/media/serializer package retains its earlier cross-compilation coverage; that does not establish other-platform Whisper support. It consumes the verified `decode` checkpoint through the canonical PCM reader. It uses the checked frontend, host encoder and decoder with independent window state; no model loader or neural subprocess is hidden. The model must have no GPU buffers, NVIDIA must be disabled before process initialisation and throughout execution, and the caller must exclude legacy model use or backend-flag changes. This stage does not accept a resident Vulkan encoder because the store cannot safely release admission while native work is retained.

The version hashes model geometry, full tokenizer, generation JSON, language/window/decoder options and suppression defaults. Caller-supplied model and runtime SHA256 values attest the loaded weight provenance and ISA/backend configuration; in-memory tensors cannot prove those identities. The caller must keep the model/tokenizer immutable during construction and execution. No default neural policy is promoted.

`asr-windows` is bounded JSONL of `windowRecord` schema 1: a dependency key plus the existing Go `WindowTranscript` fields (`Window`, `Segments`, seconds and token IDs). It is raw per-window output; overlapping text may repeat. It is not accepted by the final transcript/VTT schema. Payload and acknowledgement filenames include the complete stage dependency key and absolute window index. Every window payload is synced/published before a canonical, size/hash-bound acknowledgement is synced/published. Only an acknowledged contiguous prefix is eligible for reuse. Gaps, invalid geometry, changed hashes and noncanonical records fail closed before new inference. Unacknowledged payloads are recomputed and must match existing bytes exactly.

`TranscribePCMWindowsFrom` retains absolute timestamps and emission ownership while skipping a verified prefix. It is valid for the current independent-window decoder, which has no cross-window prior-text state. The journal avoids the store's 64-stage limit: tests cover 131 windows using one ASR stage. Individual payloads are capped at 1 MiB, the complete result at 64 MiB and the plan at 10,000 windows/four hours. Admission conservatively reserves the full result, all window payloads and acknowledgement overhead, even on retry. This can reject a near-quota retry despite reusable files; explicit storage policy is caller-owned. Journal bytes count in inventory and survive failure/reopen until job deletion.

A failed ASR stage has no completed `asr-windows` checkpoint for public download yet, but retains acknowledged windows for retry. After ASR completes, the verified raw checkpoint remains readable if a later stage fails. No final transcript, speaker labels, word alignment, overlap reconciliation or source-PTS mapping is inferred by the journal. Model quality, trained resume equivalence, long-recording performance, GPU drains and resource scheduling require separate qualification.

## Experimental Community-1 job stage

On Linux/amd64, `NewCommunity1Stage` binds an immutable caller-loaded `ExperimentalDiarization` to `diarization`. `AllowExperimental=true`, checkpoint/filter/runtime SHA256 attestations, every PCM/count/tie option, each CPU mode and a positive result cap are required. The constructor does not qualify those models or verify their provenance from in-memory weights. It selects no default modes and does not enable the lowest-index tie diagnostic implicitly. Existing neural/intermediate/tie/performance failures still apply.

The stage verifies the `decode` checkpoint, opens canonical PCM and rejects more than 128 planned windows before inference. With a 160,000-sample window and 16,000-sample step, at most 2,192,000 samples (137 seconds) fit. Increasing job duration limits does not remove this experimental model bound. File admission reserves the configured result cap plus manifest headroom; it is not CPU/RSS/GPU admission. The Go model uses synchronous CPU execution; caller model/backend ownership must cover the whole call.

The complete diarization result is one atomic checkpoint. Clustering needs all admitted windows, so a failure repeats diarization from the start and preserves earlier ASR/transcript/VTT checkpoints. Successful diarization is reused after later failure. There is no per-window Community-1 neural journal or cluster suffix resume.

`DiarizationDocument` schema 1 retains experimental status, actual PCM count, policy, window geometry, segmentation and reconstruction grids, processing path, training/cluster counts, constraint status, ambiguous-frame diagnostics and full/exclusive turns. It excludes embeddings and large intermediate arrays. Turns keep the model's raw reconstruction-frame centres, including padded-tail time; they are not clipped, rounded to integer samples, mapped to source PTS, relabelled or aligned to words. Segmentation `FirstCenter` and the postprocessor's half-frame-duration centres are distinct conventions. `NumSpeakers` overrides min/max. Count failure is retained on the single-training-row path; the clustered path returns an explicit KMeans-fallback-required error before producing a result. Gap filling can make exclusive turns overlap.

`ReadDiarizationJSON` accepts only the canonical bytes emitted by the stage, within 16 MiB. Re-encoding and geometry validation reject missing/duplicate/case-aliased/unknown keys, null lists, invalid grids/counts/paths, nonfinite turns and out-of-order or invalid speaker intervals. This is an internal checkpoint reader, not a general JSON import format. None of these checks establishes neural accuracy. It never feeds raw turns into final transcript output automatically.

## Conservative final transcript pipeline

On Linux/amd64, `NewTranscriptStage` consumes the complete `asr-windows` checkpoint and creates unlabelled `transcript` JSON. Supply the exact ASR stage version, language and window geometry. The caller must bind matching declarations: raw schema-1 windows do not carry language/generation metadata. The stage verifies each record's canonical encoding, dependency key, index and geometry, including empty windows, with 1 MiB/record, 64 MiB/stream and 100,000 raw-segment bounds.

All segments are sorted by absolute time. Only duplicate seconds, text and token IDs are coalesced. Disjoint repeated phrases are preserved. Every other temporal overlap returns `ErrTranscriptOverlap`; output-ownership intervals never discard conflicting text. Timestamps must map to integer samples within 1e-6 sample of float64 rounding error or return `ErrTranscriptSampleGrid`. The stage performs no interpolation, fuzzy matching, clipping, splitting or word alignment. Raw ASR stays readable if reconciliation fails. This is an explicit conservative subset, not general multilingual overlap qualification.

Use the ordered stage list `decode → asr-windows → transcript → vtt → diarization → speaker-transcript → speaker-vtt` when speaker output is wanted. Plain text/VTT is committed before experimental diarization, so its failure cannot erase completed transcription. An ASR-only job ends after `vtt`. Exact stage lists still apply to completed-job retries; adding stages to an already complete job is not supported.

`NewSpeakerTranscriptStage` requires explicit experimental consent and both input stage versions. It verifies matching source keys and PCM extents, rejects gap-filled turns, resolved ambiguous frames and unsatisfied non-silence count constraints. It uses full turns so an exclusive timeline cannot conceal another speaker. A cue is labelled only if exactly one real cluster covers its complete interval and no competing speaker intersects it. Exact contiguous same-speaker spans may join; positive gaps, partial coverage, speaker changes, overlaps and padded count-class IDs leave the cue unlabelled. Raw seconds are compared without widened tolerance. Text and timestamps are unchanged; no word alignment is implied.

`SpeakerTranscript` is a separate canonical document with source keys, experimental status, policy and labelled/unlabelled cue counts. `NewSpeakerVTTStage` creates a separate `speaker-vtt` checkpoint with an explicit experimental `NOTE`. It checks source provenance against the current checkpoint prefix. Neither stage overwrites the plain transcript/VTT or promotes Community-1 accuracy. Source-label validity and model accuracy still require trained/corpus qualification.

## Verification

Run `GOMAXPROCS=2 CGO_ENABLED=0 make speech-job-check`. With `ffmpeg` and `ffprobe` on PATH, run `make speech-job-media-integration` for three repetitions of the real WAV/AAC job tests.

Tests cover upload publication, retry without repeated ASR, transcript retention after later failure, exact configuration/version dependencies, payload corruption, orphan conflicts, quota/metadata rejection, independent-process locking, cancellation, progress/stage panics, concurrent read-only downloads, explicit deletion and incomplete-upload inventory.

Tests also cover fixed-header PCM identity, decode-config changes, failed adapter metadata, scratch accounting, retained-scratch admission, cancellation drains, JSON shape, WebVTT quoting/escaping and transcript/VTT reuse after later failure. Explicit real-FFmpeg tests use generated WAV and AAC/M4A only; repeated PCM checkpoints are byte-identical. AAC padding is retained in the measured sample count.

Unix child-process tests SIGKILL actual workers after payload fsync, payload publication, manifest fsync and manifest rename, plus two decoder scratch boundaries using a fixture adapter and two Whisper window payload/acknowledgement boundaries using synthetic inference. Community-1 tests add three whole-stage result/payload/manifest kill boundaries. They verify preserved ASR, recomputation before final publication and no repeated diarization after its checkpoint is committed. Reduced checked-in Community-1 fixtures exercise actual Go silence, single-row and clustered paths, direct-output equality, fresh-job determinism and cancellation/retry; no trained weights run. The actual Go frontend/encoder/decoder adapter is checked separately with two-channel zero-layer toy tensors; these tests establish wiring, not speech quality. Reconciliation tests cover exact duplicates, conflicting text/tokens/timing, missing/reordered/oversized records, sample-grid rejection, speaker coverage/gaps/overlaps/padded classes, provenance, separate plain/speaker output and three further process-kill boundaries. They exercise recovery without Go defers; they do not simulate power removal or storage hardware faults. Architecture cross-compilation does not establish filesystem behaviour on another OS.
