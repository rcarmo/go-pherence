# Integrated Whisper and Community-1

The speech port is being implemented in go-pherence on `feat/speech-simd-vulkan`, starting from `d08ce322292847a7978dcace1cc73a50d0bf8450`. Reference: Rui's **message:1671 / attachment:167**, implementation-sidebar request of 11 September 2026, and the subsequent clarification to use FFmpeg temporarily. Go owns inference; high-performance Go/Plan 9 assembly SIMD and Vulkan are first-class backends.

## Current implementation

- `loader/audio/media`: a separate, temporary FFmpeg/ffprobe adapter. Probe and decode accept context cancellation and return format/sample metadata. Conversion produces mono 16 kHz signed 16-bit RIFF/WAV without deleting the input or replacing an existing destination.
- `loader/audio.WhisperLogMel`: checked exact-contract 80/128-bin features, FFT400 geometry, periodic Hann, centred reflect padding, Slaney filters, final-frame removal and Whisper normalisation. The current DFT uses the existing checked SIMD `Ddot`; the mixed-radix Plan 9 FFT optimisation has not been implemented.
- `models/whisper.MelFlatFromSamplesChecked`: model-configured entry point to those features. Existing inference/CLI defaults remain unchanged pending real-checkpoint multilingual/timestamp qualification. The legacy 512-point/GPU mel path remains a separate implementation and is not the exact oracle.
- Four new synthetic 128-band fixtures generated independently with checksum-pinned Transformers 4.57.1 numerical functions; absolute tolerance `1e-5`. Existing 80-band reference still passes. No model weights are used by these tests.
- `media.OpenCanonicalPCM`: a validated canonical-WAV reader using positional reads and caller-owned float32 output. It keeps one descriptor and a 16 KiB conversion buffer, rather than loading a whole recording. It does not invoke FFmpeg or import model/video code.
- `models/whisper.WindowPlan`: on-demand integer-sample windows, disjoint emission-ownership intervals and explicit final-window padding. New code can plan all four hours without the historical 100-chunk cutoff. Legacy chunked inference remains unchanged. The opt-in `TranscribePCMWindows` path now connects this reader/planner to checked features and the existing Go encoder/decoder; real-checkpoint quality is unqualified.

This is a foundation checkpoint. Community-1 porting, the model-ready Vulkan execution layer, new assembly kernels, the integrated job service, real-checkpoint quality tests and performance targets are not complete. Neither existing service has been restarted or deployed from this branch.

## Ownership and boundaries

| Owner | Responsibility |
|---|---|
| `loader/audio/media` | Temporary FFmpeg adapter and future pure-Go adapter; output PCM frame count and media contract |
| `loader/audio` | Exact Whisper/WeSpeaker model-specific features; canonical PCM input |
| `models/whisper` | Encoder/decoder, model formats, language/task policy, state and word timestamps |
| `models/speaker/community1` (planned) | SincNet/BiLSTM/powerset, WeSpeaker/ResNet, masked pooling, PLDA/VBx and global full/exclusive turns |
| `backends/simd/runtime`, existing FFT/half owners | Checked dispatch and reusable Go/Plan 9 assembly microkernels; scalar oracles and ISA fallbacks |
| `backends/vulkan` | Device/features/FFI correctness, typed arenas, owned command/descriptor/fence state, resident multi-op graphs and shaders |
| `cmd/audio` and a narrow job package (planned) | CLI/HTTP integration, resource scheduling, durable uploads, checkpoints, retry/cancel and VTT/JSON |
| `rcarmo/go-264/audio/...` (independent workstream) | Public pure-Go audio container/codec/resampler/DSP library; audio-only users must not depend on video or go-pherence |

Do not move codec ownership into go-pherence or wait for go-264 before developing the neural pipeline. Keep the `media.Adapter` contract narrow so a separately qualified go-264 backend can replace FFmpeg later. Do not conceal whisper.cpp/PyTorch/ONNX inference underneath the final Go service. Offline model/fixture conversion and the temporary media subprocess are explicitly separate from inference.

## Media contract and limits

`NewFFmpeg` requires configured executable paths. Zero limits receive defaults; negative limits and durations above four hours are rejected. Default input limit is 512 MiB; canonical output is bounded by four hours of s16 mono samples plus header allowance. Probe output and retained stderr are bounded; errors do not echo subprocess payloads.

Only local regular files with WAV or ISO BMFF signatures and supported extensions are accepted (`.wav`, `.m4a`, `.mp4`, `.mov`). This deliberately excludes RF64, unusual MOV files lacking an initial `ftyp`, playlists and remote URLs. The lowest valid audio stream index is chosen. FFmpeg gets explicit demuxers, `file,pipe` protocols, disabled MOV external references/absolute aliases, no video/subtitle/data mapping, and two threads. The caller must place inputs and outputs in its private job directory and keep the input immutable for the operation; this library is not an untrusted multi-user filesystem sandbox.

Probe has a 30-second deadline; decode has an eight-hour deadline, both shortened by the caller's context. A trusted injected `Runner` must honour that contract and wait for its owned process. Decode writes to a unique mode-0600 same-directory temporary file, monitors its size, uses FFmpeg's coarse `-fs` ceiling, and strictly validates RIFF structure, channel/rate/encoding, data alignment and frame limits before publishing. Hitting the byte ceiling is rejected, even if FFmpeg exits successfully. The monitor can overshoot its limit by subprocess buffering/scheduling before termination; this is bounded-output checking, not an OS-level memory/disk quota. The eventual service must enforce per-worker memory/disk admission too.

No duration-based `-t` truncation is used to make a long file appear valid. Actual decoded frame count comes from the output WAV data chunk. The probe timeline is an estimate from reported duration. It does not certify exact gapless sample accounting, original track PTS mapping or edits for every container; those require a dedicated fixture matrix. Known fixtures include 44.1/48 kHz WAV and a short synthetic AAC/M4A. No private recording was read for these tests.

Publication is no-clobber: Linux `renameat2(RENAME_NOREPLACE)`, portable same-directory link/unlink fallback. Atomic visibility is tested; power-loss/fsync durability is not yet provided by this adapter and belongs in the job checkpoint layer. The input is preserved on success, failure and cancellation; the job layer owns its retention policy.

### Bounded PCM consumption and window planning

`OpenCanonicalPCM(ctx, path)` validates and reads the same open file. The WAV scanner bounds metadata chunks to 4096 and checks cancellation between chunks. The caller must provide an immutable regular file in a trusted directory; this is not a hostile-filesystem sandbox. `ReadSamplesAt(ctx, dst, startFrame)` converts little-endian s16 to float32 in 8192-frame blocks and leaves `dst[n:]` untouched. Reads and close are serialised; waiting for another read can be cancelled. An underlying regular-file syscall already in progress is not interruptible by context. Unexpected short reads within the validated data extent are errors, not normal EOF.

Generic WAV byte limits are inclusive. FFmpeg publication separately rejects output at its `-fs` ceiling because reaching that ceiling can mean a successful but truncated decode. Both behaviours have regression tests; a standalone four-hour sparse WAV is accepted and its final frame can be read without loading the recording.

`NewWindowPlan(total, length, overlap)` accepts explicit canonical-PCM sample counts. `At(index)` derives a window without allocating a list; `ReadWindow` uses caller-owned scratch and zero-pads only the final missing tail. Overlap ownership is split at the midpoint so emission intervals cover the recording exactly once, with no gaps. This describes eligibility for future timestamp/word merging; it does not perform transcript deduplication, source-container PTS mapping or VAD packing. The caller enforces the overall job limit and model padding requirements.

Tests cover arbitrary read positions, signed endpoints, EOF versus truncation, concurrent readers, cancellation while waiting, idempotent close, four-hour sparse metadata, maximum-int64 window arithmetic, exact/odd overlap boundaries, final-one-frame padding, and reader → window → checked-feature composition on synthetic PCM. Isolated reader and window operations allocate zero times with preallocated scratch in the tested Go build; feature generation still allocates its result and DFT workspace. These are functional/allocation checks, not throughput measurements.

## Exact frontend contract

`WhisperLogMel(samples, bands)` accepts 160..480000 mono 16 kHz float samples, `bands=80` or `128`, and rejects non-finite values. It does not resample or right-pad. It returns channel-major `[bands, floor(samples/160)]` output. Window planning, padding to model geometry and mapping words to original sample indices remain caller responsibilities. Do not pass a whole recording to one feature call.

The `WhisperLogMel80` compatibility API delegates to the new implementation; invalid inputs return no features. MOSS uses the 80-bin path on its already-padded windows. No model defaults or weight formats have been silently changed.

`loader/audio/testdata/whisper_logmel128_transformers_4_57_1.json` contains synthetic broadband, impulse/odd-boundary, short-reflect and silence cases. `scripts/whisper_logmel_reference.py` uses NumPy and only six checksum-pinned upstream numerical definitions. It does not import Transformers/PyTorch or load a checkpoint. See fixture metadata for source URL/hash and NumPy version. This reference is the NumPy Transformers frontend; exact Torch/whisper.cpp numerical differences still need separate real-model qualification.

## Opt-in PCM inference

`Whisper.TranscribePCMWindows(ctx, source, totalSamples, tokenizer, options, emit)` accepts the canonical PCM reader and actual decoded sample count. It reuses window-sized audio scratch, computes the checked frontend, calls the Go encoder and creates fresh decoder state per window. It emits synchronously instead of accumulating a full recording or transcript. It rejects jobs over four hours, overlap above half the analysis window, and plans exceeding 10000 windows. It never silently cuts off at the legacy 100-window boundary.

The caller supplies an immutable model/tokenizer, explicit language code and original-language transcription task. Validation checks model geometry and required tensor lengths before execution. The new decoder verifies control/timestamp IDs against the tokenizer. The legacy constants match large-v3/turbo but are one ID too high for older multilingual vocabularies. [Pinned tokenizer/generation contracts](speech-generation-manifest.json) cover tiny and turbo; both public tokenizers were tested without model weights. The legacy APIs remain unchanged.

The checked timestamp loop enforces initial bounds, pairing, monotonicity and aggregate timestamp probability using the Transformers 4.57.1 rule contract. It counts the three-token prompt within the decoder position limit. Invalid logits, exhausted token budgets and text without a closing timestamp return errors. It does not invent a 30-second segment end. `MaxInitialTimestampIndex` is explicit: zero forces 0.00; the pinned configurations use 50 (one second).

Callbacks receive window metadata and segments in canonical PCM seconds, clipped to the actual audio tail. These are raw per-window results: overlapping windows may repeat text. Word alignment, overlap reconciliation, automatic language detection, no-speech thresholds, temperature/quality fallback, prior-text conditioning and source-container PTS mapping are not implemented. Do not publish the callbacks directly as a qualified final VTT.

Checked calls are serialised because existing Whisper kernels have package-level state. Callers must also exclude concurrent legacy execution and avoid callback re-entry. Cancellation is checked around stages and decoder tokens; an encoder or cross-KV operation already running cannot yet be interrupted. Failed windows never reach the callback; previously emitted windows remain available for caller-owned checkpoints.

Tests include scripted-logit state transitions, 131 synthetic windows with reused scratch, final padding, callback/stage errors, malformed model/tokenizer rejection, and a two-channel zero-layer toy tensor pipeline through the actual feature/encoder/decoder code. Toy tensor execution proves wiring only. No real checkpoint, private recording, GPU, new assembly kernel, performance benchmark or service was run. Two narrow delegated reviews of generation/wiring timed out without findings; no independent approval was obtained. Parent review and focused tests passed.

### Checked tensor loading

`LoadModelSourceChecked(ctx, source, cfg)` loads the opt-in model from a `CheckedTensorSource`; `safetensors.File` implements that contract. It validates the complete encoder/decoder inventory, exact HF tensor shapes, F32/F16/BF16 dtypes, byte extents and non-overlap before the first weight read. Decoder FFN dimensions are checked independently from encoder FFN dimensions. Missing parameters, extra tensors, integer/quantised dtypes, key-projection biases and untied output projections are rejected. An optional `proj_out.weight` must equal the widened token embedding exactly.

Loaded weights are copied into model-owned slices and checked for finite values; the caller can close the source immediately after loading. Peak memory includes retained weights plus source and owned buffers for the current tensor. The caller must admit that memory before loading. Context is checked around source calls and during copy/validation loops; a source call already in progress is not interruptible. The source owns header/file bounds and must expose one byte-offset address space. Sharded sources require offset normalisation first.

The loader creates no GPU allocations, global packing caches or inference workers. It does not automatically import a generation configuration or model config JSON; callers still own those validated contracts and decoder suppression settings. It does not replace legacy `LoadModel` or `LoadEncoderSource`. Synthetic fixtures test all bindings, reject malformed metadata before payload reads, exercise genuine F32/F16/BF16 safetensors widening and verify model ownership after file closure. Real checkpoint tensors, GGML/quantised loading and model-output parity are not qualified. The loader checkpoint's [verification](../benchmarks/speech-foundations/load-verification.json) records 47 passing test/subtest events with zero failures/skips, passing focused make/safetensors/vet/audio builds and unchanged full-tree baseline errors. A narrow delegated review returned no blocking finding.

## Frozen references and proposed acceptance

`speech-reference-manifest.json` records the analysed sources, installed reference weight hashes and historical comparisons. The source of the old performance numbers is the separately deployed `projects/whisper-stt` measurement record. They are historical targets; no Go throughput result exists yet.

- Freeze original-language transcription (`task=transcribe`), tokenizer/model/generation config, complete audio coverage, VAD/window geometry and quality tolerances before speed tuning.
- Initial target: at least 8× realtime ASR and 5× combined ASR+diarization, with at least 1.5× faster standalone speaker inference on matched fixtures.
- Proposed maximum WER and corpus DER regression: 0.5 percentage points absolute, assessed with paired uncertainty by language/corpus. These budgets require ratification; the tiny public diarization fixture's score is not a universal quality target.
- Cross-backend token identity is diagnostic; deterministic state/timeline invariants and complete task-quality checks are acceptance gates.
- Use licensed public/synthetic fixtures. Access to a private long recording requires explicit user permission; no identifying file names or transcript contents belong in committed evidence.
- Feature, media, SIMD and Vulkan tests must expose unsupported paths/fallbacks. Shape-specific Plan 9 assembly requires reference comparison, ABI/tail/ISA guards, disassembly and meaningful measured benefit; no hidden BLAS/CGo inference.

## Implementation sequence

1. Review the media/feature checkpoint and complete the long-file media/timebase fixture matrix.
2. Wire the checked frontend into an explicit original-language speech session, with token/model config validation and non-truncating window coverage. Keep legacy behaviour until qualified.
3. Freeze and validate Community-1 graph/weight contracts, then scalar/Go stage oracles. ECAPA is not a substitute.
4. Add per-session CPU workspace and Plan 9 assembly improvements for measured decoder/attention/convolution/recurrent/DSP shapes.
5. Repair Vulkan FFI/features, resource lifetime and submission ownership; add resident encoder and Community-1 kernels, then compare hybrid versus full GPU placement.
6. Implement durable jobs, backend/config-keyed checkpoints, independent transcript download, retries, cancellation and resource coordination.
7. Qualify under an agreed hardware window against matched old runtime controls; do not disturb existing LLM work.
8. Integrate the independently qualified go-264 adapter and repeat media/quality/performance gates before removing FFmpeg from serving.

## Verification commands

```sh
# No model weights, GPU initialisation, service or benchmark required.
GOMAXPROCS=2 CGO_ENABLED=0 make speech-foundations-check
# Optional pinned public tokenizer/config checks; see speech-generation-manifest.json.
GO_PHERENCE_TEST_TOKENIZER_DIR=/path/to/pinned-files GOMAXPROCS=2 CGO_ENABLED=0 \
  go test -p=1 ./models/whisper -run TestCheckedTimestampPinnedTokenizers
# Short synthetic media only; requires ffmpeg and ffprobe installed.
GOMAXPROCS=2 CGO_ENABLED=0 make speech-media-integration
```

Set `TMPDIR` and `GOTMPDIR` to an existing writable directory if `/workspace/tmp` is unavailable. Use Go 1.26.2 as required by `go.mod`.

The initial `651489e` checkpoint passed the focused make targets, affected-package vet, all audio-command builds and additional existing CPU convolution/LayerNorm/attention reference tests. The JSON record counts 54 passing test/subtest events and one intentionally skipped opt-in FFmpeg test; that FFmpeg test then passed in its separate enabled target. This is not a count of 54 top-level tests. No models, GPU calls or service changes occurred. See [verification evidence](../benchmarks/speech-foundations/verification.json).

`go build ./...` failed in the unrelated `backends/spacemit/aicpu/aipool`, `cmd/diffusiongemmainspect` and `cmd/diffusiongemmaserve` packages. A clean archive of untouched `d08ce322` produced identical errors; [baseline error listing](../benchmarks/speech-foundations/full-build-known-errors.txt) is retained. Those source files were not modified. Full model/backend suites and race tests have not been run. A delegated read-only review timed out without findings; parent source review and the listed tests were completed. The entire speech port and speed targets are not complete.

The subsequent reader/window checkpoint is recorded separately in [streaming verification](../benchmarks/speech-foundations/stream-verification.json) so the earlier counts remain historical. A narrowly scoped delegated source review flagged inclusive byte-limit semantics; this was resolved by separating generic reader limits from FFmpeg's cap-reached rejection. No other blocking finding was reported in the three reviewed files, under the documented immutable-file assumptions. Race-detector tests and model-quality/throughput runs remain unperformed.

The opt-in PCM inference checkpoint has separate [verification](../benchmarks/speech-foundations/pcm-verification.json): 33 passing test/subtest events, zero failures, two legacy asset-dependent tokenizer skips. Pinned public tiny/turbo tokenizer checks passed. Focused make targets, affected vet/audio builds and the exact full-tree baseline-error comparison passed. No real checkpoint or performance workload ran.
