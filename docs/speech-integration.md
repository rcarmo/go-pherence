# Integrated Whisper and Community-1

The speech port is being implemented in go-pherence on `feat/speech-simd-vulkan`, starting from `d08ce322292847a7978dcace1cc73a50d0bf8450`. Reference: Rui's **message:1671 / attachment:167**, implementation-sidebar request of 11 September 2026, and the subsequent clarification to use FFmpeg temporarily. Go owns inference; high-performance Go/Plan 9 assembly SIMD and Vulkan are first-class backends.

## Current implementation

- `loader/audio/media`: a separate, temporary FFmpeg/ffprobe adapter. Probe and decode accept context cancellation and return format/sample metadata. Conversion produces mono 16 kHz signed 16-bit RIFF/WAV without deleting the input or replacing an existing destination.
- `loader/audio.WhisperLogMel`: checked exact-contract 80/128-bin features, FFT400 geometry, periodic Hann, centred reflect padding, Slaney filters, final-frame removal and Whisper normalisation. The current DFT uses the existing checked SIMD `Ddot`; the mixed-radix Plan 9 FFT optimisation has not been implemented.
- `models/whisper.MelFlatFromSamplesChecked`: model-configured entry point to those features. Existing inference/CLI defaults remain unchanged pending real-checkpoint multilingual/timestamp qualification. The legacy 512-point/GPU mel path remains a separate implementation and is not the exact oracle.
- Four new synthetic 128-band fixtures generated independently with checksum-pinned Transformers 4.57.1 numerical functions; absolute tolerance `1e-5`. Existing 80-band reference still passes. No model weights are used by these tests.
- `media.OpenCanonicalPCM`: a validated canonical-WAV reader using positional reads and caller-owned float32 output. It keeps one descriptor and a 16 KiB conversion buffer, rather than loading a whole recording. It does not invoke FFmpeg or import model/video code.
- `models/whisper.WindowPlan`: on-demand integer-sample windows, disjoint emission-ownership intervals and explicit final-window padding. New code can plan all four hours without the historical 100-chunk cutoff. Legacy chunked inference remains unchanged. The opt-in `TranscribePCMWindows` path now connects this reader/planner to checked features and the existing Go encoder/decoder; real-checkpoint quality is unqualified.

- `models/speaker/community1.Powerset`: bounded hard/soft powerset-to-local-speaker conversion and score-column permutation mapping, checked against pyannote 4.0.7. It is not yet connected to a segmentation model.
- `models/speaker/community1.LSTM`: owned, checked unprojected IFGO LSTM stack, including bidirectional chronological outputs, per-layer intermediates and terminal states. Scalar and existing Plan 9 SIMD GEMV paths match synthetic PyTorch fixtures. Checkpoint loading and the complete segmentation graph are not connected.

- `models/speaker/community1.SincNet`: experimental 16 kHz frontend with learned even/odd sinc filters, convolution/pooling/instance-norm stages and frame-grid metadata. Its narrow-band float32 oracle gate fails; it is not qualified or connected to a segmentation model.

This is a foundation checkpoint. Community-1 porting, the model-ready Vulkan execution layer, new assembly kernels, the integrated job service, real-checkpoint quality tests and performance targets are not complete. Neither existing service has been restarted or deployed from this branch.

## Ownership and boundaries

| Owner | Responsibility |
|---|---|
| `loader/audio/media` | Temporary FFmpeg adapter and future pure-Go adapter; output PCM frame count and media contract |
| `loader/audio` | Exact Whisper/WeSpeaker model-specific features; canonical PCM input |
| `models/whisper` | Encoder/decoder, model formats, language/task policy, state and word timestamps |
| `models/speaker/community1` | Powerset component started; SincNet/BiLSTM, WeSpeaker/ResNet, masked pooling, PLDA/VBx and global full/exclusive turns still planned |
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

The checked timestamp loop enforces initial bounds, pairing, monotonicity and aggregate timestamp probability using the Transformers 4.57.1 rule contract. It counts the three-token prompt within the decoder position limit. Invalid logits, exhausted token budgets and text without a closing timestamp return errors. It does not invent a 30-second segment end. Without a generation policy, `MaxInitialTimestampIndex` is explicit: zero forces 0.00. A supplied checked policy owns that bound; the pinned configurations use 50 (one second).

Callbacks receive window metadata and segments in canonical PCM seconds, clipped to the actual audio tail. These are raw per-window results: overlapping windows may repeat text. Word alignment, overlap reconciliation, automatic language detection, no-speech thresholds, temperature/quality fallback, prior-text conditioning and source-container PTS mapping are not implemented. Do not publish the callbacks directly as a qualified final VTT.

Checked calls are serialised because existing Whisper kernels have package-level state. Callers must also exclude concurrent legacy execution and avoid callback re-entry. Cancellation is now checked between frontend frames, encoder operators, cross-KV projections/reorders and decoder tokens. A running kernel, fused operator, allocation or reorder still finishes before cancellation is reported; no wall-clock cancellation deadline is qualified. Failed windows never reach the callback; previously emitted windows remain available for caller-owned checkpoints.

Tests include scripted-logit state transitions, 131 synthetic windows with reused scratch, final padding, callback/stage errors, malformed model/tokenizer rejection, and a two-channel zero-layer toy tensor pipeline through the actual feature/encoder/decoder code. Toy tensor execution proves wiring only. No real checkpoint, private recording, GPU, new assembly kernel, performance benchmark or service was run. Two narrow delegated reviews of generation/wiring timed out without findings; no independent approval was obtained. Parent review and focused tests passed.

### Cooperative cancellation

`WhisperLogMelContext` and `MelFlatFromSamplesCheckedContext` preserve the exact frontend operations with checks between DFT frames and every 16384 elements in input/output validation loops. The one-time lookup-table initialisation and reflect-padding copy remain synchronous. `Encoder.ForwardContext` checks between convolution, normalisation, projection, attention and FFN operators, plus bounded residual/position loops. `NewDecoderStateContext` checks between layer allocations, cross-K/V projections and head-major reorders. The checked PCM path calls all three context-aware stages. Existing non-context wrappers keep their signatures and operation order.

Cancellation returns no partial features, encoder output or decoder state. A completed earlier window callback remains valid; failed windows are not emitted. Existing kernels retain their synchronous worker joins. Interruptible tiled kernels, device-loss handling, GPU resource cancellation and per-token decoder-internal checkpoints still require separate implementation and testing. Source/model ownership, validation and exclusive-execution requirements are unchanged.

Synthetic fault injection exercises 46 encoder checkpoints, 13 cross-KV checkpoints and 100 whole-PCM checkpoints without relying on sleeps. Pre-cancel checks run before touching malformed model inputs; cancellation at an encoder observer prevents subsequent operators; a fresh empty PCM call verifies the serial gate is released after each injected cancellation. Numerical comparisons with the pre-change `3f5a91b` frontend/encoder/decoder-state functions are bit-exact for tiny synthetic fixtures, including observer boundaries and the first decoder step. No real checkpoint or cancellation-latency benchmark ran. A narrow delegated diff review returned no blocking finding. See [cancellation verification](../benchmarks/speech-foundations/context-verification.json).

### Checked tensor loading

`LoadModelSourceChecked(ctx, source, cfg)` loads the opt-in model from a `CheckedTensorSource`; `safetensors.File` implements that contract. It validates the complete encoder/decoder inventory, exact HF tensor shapes, F32/F16/BF16 dtypes, byte extents and non-overlap before the first weight read. Decoder FFN dimensions are checked independently from encoder FFN dimensions. Missing parameters, extra tensors, integer/quantised dtypes, key-projection biases and untied output projections are rejected. An optional `proj_out.weight` must equal the widened token embedding exactly.

Loaded weights are copied into model-owned slices and checked for finite values; the caller can close the source immediately after loading. Peak memory includes retained weights plus source and owned buffers for the current tensor. The caller must admit that memory before loading. Context is checked around source calls and during copy/validation loops; a source call already in progress is not interruptible. The source owns header/file bounds and must expose one byte-offset address space. Sharded sources require offset normalisation first.

The tensor loader creates no GPU allocations, global packing caches or inference workers. It accepts an already validated model config. The configured loader below imports model/generation JSON and returns a per-call policy. It does not replace legacy `LoadModel` or `LoadEncoderSource`. Synthetic fixtures test all bindings, reject malformed metadata before payload reads, exercise genuine F32/F16/BF16 safetensors widening and verify model ownership after file closure. Real checkpoint tensors, GGML/quantised loading and model-output parity are not qualified. The loader checkpoint's [verification](../benchmarks/speech-foundations/load-verification.json) records 47 passing test/subtest events with zero failures/skips, passing focused make/safetensors/vet/audio builds and unchanged full-tree baseline errors. A narrow delegated review returned no blocking finding.

### Model and generation JSON

`ParseModelConfigChecked` imports bounded HF geometry and requires multilingual Whisper, GELU, unscaled embeddings and the expected token layout. `ParseGenerationConfigChecked` validates generation IDs, the complete language map, task IDs, suppression lists, initial timestamps and the position budget against that model and tokenizer. Both reject duplicate JSON keys, trailing input, null typed members, nesting above 32 and input larger than 1 MiB. Unknown fields are rejected; accepted training/provenance fields in model config have no inference effect. Configurations outside the supported field set need an explicit extension.

`LoadConfiguredModelSourceChecked(ctx, source, modelJSON, generationJSON, tokenizer)` validates both documents before tensor payload reads. It returns the owned model and an immutable `CheckedGenerationConfig`. Pass the latter in `PCMTranscribeOptions.Generation`; it takes precedence over decoder suppression lists without changing shared decoder state. A per-call `MaxNewTokens` can shorten the policy budget, while a nonzero per-call `MaxInitialTimestampIndex` is rejected when a policy is supplied. For both pinned models the policy imports 448 positions including the prompt, initial timestamp index 50 and begin-suppression `[220,50257]`. Turbo's model config has a different begin-suppression list; the generation config takes precedence.

The checked path explicitly uses the requested language, `transcribe` and timestamps. It validates but overrides source `forced_decoder_ids`, source language/task defaults and `return_timestamps`; it does not append the old forced prompt. Non-default sampling, beam, repetition and length-penalty controls fail. HF's default temperature 1 is accepted only without sampling; selection is greedy argmax. Alignment-head metadata is checked but word alignment is not implemented. Legacy APIs retain their existing defaults.

Pinned tiny/turbo `config.json` files now have revision URLs and hashes in the [generation manifest](speech-generation-manifest.json), alongside tokenizers and generation JSON. Public-metadata tests verify the imported configs against the named Go configs. Synthetic tests verify rejection before tensor reads, effective suppression in token selection and immutable per-call policy through the toy PCM pipeline. A delegated parser review timed out; parent review and the [config verification checks](../benchmarks/speech-foundations/config-verification.json) passed. Real checkpoint inference, quality and performance remain unqualified.

## Community-1 powerset component

`NewPowerset(speakers, maxActive)` enumerates local-speaker subsets by cardinality, then lexicographic combinations. For `(3,2)` the classes are `{}`, `{0}`, `{1}`, `{2}`, `{0,1}`, `{0,2}`, `{1,2}`. Geometry must come from the segmentation checkpoint; the constructor does not identify a checkpoint or infer a speaker count. It accepts 1..8 local slots and up to 4096 segmentation frames per call. These frame indexes are not PCM sample indexes.

`Decode(ctx, scores, frames, PowersetHard)` chooses the first maximum class and emits its binary speaker membership. `PowersetSoft` sums `exp(log_probability)` over classes containing each speaker; it does not apply another softmax. Normalisation belongs to the caller. Both reject malformed shapes, NaN/+Inf and all-impossible rows, allow `-Inf` for individual impossible classes, and return no partial result on cancellation. Soft mode also rejects positive log probabilities. Outputs are owned; the model does not mutate input scores.

`ClassPermutation(slots)` returns powerset-column indexes for `output[:,j] = input[:,slots[j]]` in local-speaker space. It does not perform assignment or clustering. Hard argmax ties follow the permuted column order and therefore need not commute with permutation. No thresholds, VAD, timestamps, global identities or exclusive turns are produced here. Existing ECAPA and energy-VAD code is not used as a Community-1 substitute.

The installed pyannote 4.0.7 source matched commit `b749285c5cdd4636b2edc7f766f1352c8dde9369` at SHA-256 `7eeb5691c20337bd24462f6a4ee2e56e279564bfef7875776df0a41f245b0325`. [Fixture metadata](../models/speaker/community1/testdata/powerset-reference.json) pins the source and runtime; [NOTICE](../models/speaker/community1/NOTICE) preserves MIT attribution. The offline [generator](../scripts/community1_powerset_reference.py) loads only that source file and applies PyTorch CPU operations to tiny synthetic arrays on one thread. It loads no weights, audio or neural pipeline. Go runtime inference has no PyTorch dependency.

Four geometries `(1,1)`, `(3,2)`, `(4,2)` and `(4,4)` match hard outputs exactly and soft outputs within `1e-6`, including all 55 reference permutations. Safety tests cover ties, overlapping classes, invalid arrays, maximum bounds, ownership and deterministic per-row cancellation. A narrow delegated review found no blocking issue. The complete SincNet/BiLSTM segmentation graph, masked WeSpeaker embeddings, PLDA/AHC/VBx, full/exclusive turn reconstruction, specialised SIMD kernels and Vulkan acceleration remain unimplemented in this package. No diarization quality or speed was measured. [Powerset verification](../benchmarks/speech-foundations/powerset-verification.json) records the four top-level tests, oracle geometry coverage, regression/build checks and arm64 cross-build (not executed).

## Community-1 recurrent component

`NewLSTM(ctx, cfg, layers)` checks all PyTorch IFGO weight/bias shapes and finite values before copying them into an immutable model. It supports one batch, 1..512 input features, 1..256 hidden units, 1..4 layers and optional bidirectionality. Both input and recurrent biases are preserved. There is no projection, training dropout or implicit checkpoint geometry. `Forward` accepts 1..4096 complete segmentation frames, with zero or explicitly supplied initial hidden/cell state.

The output concatenates forward then reverse features at the same chronological frame. Hidden/cell results are ordered by layer then direction; reverse terminal state is reached after frame zero. The bidirectional path needs the complete window and cannot be resumed causally by splitting it. Unidirectional split/resume is tested. Inputs, source weights and returned results do not alias retained model state. Per-layer observers expose transient read-only views; copy them to retain. Independent calls own their scratch and recurrent state.

`LSTMScalar` is the float32 reference. `LSTMSIMD` dispatches matrix-vector projections through existing checked `simd.GemvRows` and its Plan 9 assembly/fallbacks. Activations use Go scalar exp/tanh. No specialised recurrent assembly was added and no speedup was measured. Two alternating sequence buffers and fixed gate/state scratch avoid per-frame allocations. The measured allocation count is seven per call for the tested two-layer case, independent of one versus seven frames. Cancellation is checked around projections and recurrent steps; running GEMV/callbacks are synchronous. Errors return no partial result, although an observer may already have received completed layers.

Six [synthetic PyTorch fixtures](../models/speaker/community1/testdata/lstm-reference.json) cover uni/bidirectionality, one to three layers, odd widths, nonzero initial state and sequences up to 65 frames. Both Go modes match every layer output, full sequence and terminal hidden/cell states within `2e-6`. A separate three-frame 60-input/128-hidden/two-layer synthetic case checks scalar/SIMD parity at representative PyanNet constructor dimensions; it is not checkpoint validation. The [offline generator](../scripts/community1_lstm_reference.py) pins PyanNet and Torch RNN source hashes, records the Torch build commit, runs CPU/one thread with MKLDNN disabled and loads no trained weights or audio. Regeneration produces identical JSON.

Tests cover bad shapes/modes/states, NaN/Inf, affine overflow, source/output ownership, state reset, concurrent immutable-model calls, per-layer cancellation and fixed allocation counts. Deterministic fault injection reaches 78 scalar and 58 SIMD forward checkpoints in the small bidirectional fixture. A narrow delegated source review found no blocking issue. The [verification record](../benchmarks/speech-foundations/lstm-verification.json) separates these component checks from unmeasured model quality and performance. The SincNet qualification gap below, segmentation head/log-softmax, actual checkpoint binding and the remaining Community-1 pipeline still require work.

## Experimental SincNet frontend — qualification failed

`NewSincNet` owns finite learned cutoffs, affine normalisation and convolution weights. At 16 kHz, 40 learned band pairs produce 80 even/odd filters of length 251 using the pinned asteroid-filterbanks 0.4.0 contract. The forward path follows pyannote SincNet: whole-waveform instance norm; sinc convolution and absolute value; then three max-pool/affine-instance-norm/leaky-ReLU stages, with 80→60 and 60→60 kernel-5 convolutions between them. Population variance uses epsilon `1e-5`. Scalar Go and existing Plan 9 `Sdot` modes are available; no specialised convolution assembly or throughput result was added.

`Grid` derives frame counts, step, first sample-index centre and nominal receptive-field size. For stride `s`, the step is `27*s`, first centre `125+37*s`, and nominal receptive field `251+74*s` samples. Input windows must leave at least two positions for every instance norm and stay within 160000 input samples and 4096 output frames. Because instance normalisation uses the entire window, the nominal convolution receptive field is not a strict dependency bound. Arbitrary streaming chunks would change the result. `Forward` returns owned frame-major `[frames,60]` values suitable for the LSTM input layout, but the modules are not wired together yet.

Seven pinned synthetic fixtures include two narrow-band waves plus impulse, silence, constant and two broadband cases. For the five non-narrow-band cases, both modes pass every boundary and final output at `2e-4` absolute tolerance; filter coefficients pass `2e-6`. Narrow-band output errors exceed the unchanged `2e-4` gate:

| Stride | Scalar max absolute error | Existing SIMD max absolute error |
|---|---:|---:|
| 10 | 0.00761348009 | 0.00461539626 |
| 1 | 0.00120222569 | 0.00226772483 |

The [strict gate log](../benchmarks/speech-foundations/sincnet-strict-gap.txt) retains all four failures. The ordinary suite explicitly skips that gate; passing development tests do not qualify this frontend. Reproduce it with:

```sh
GO_PHERENCE_TEST_SINCNET_STRICT=1 GOMAXPROCS=2 CGO_ENABLED=0 \
  go test -p=1 -count=1 ./models/speaker/community1 -run TestSincNetStrictNarrowBandOracle
```

Diagnostics using exact oracle filter coefficients still differ after convolution/normalisation. A separate Torch float64 calculation using those coefficients differs from Torch float32 by about `0.0065` for the stride-10 narrow-band case, whose first-stage minimum channel variance is about `3e-11`. Matching Torch's float32 affine scale/shift/FMA ordering improved results; it did not close the gate. The remaining discrepancy needs kernel/reduction analysis and actual model-level quality evidence before integration, not an automatic tolerance increase.

Boundary/ownership tests and a narrow delegated source review found no additional blocking defect. Cancellation checks cover stages, channels and bounded loops; running scalar/SIMD dots and observers are synchronous. There is no model or input mutation. The compressed [fixtures](../models/speaker/community1/testdata/sincnet-reference.json.gz) and [offline generator](../scripts/community1_sincnet_reference.py) pin pyannote/filterbank sources, use only synthetic weights/PCM and regenerate byte-identically. The [verification record](../benchmarks/speech-foundations/sincnet-verification.json) keeps passing development checks separate from failing qualification. No trained model, GPU, private audio, service change or production default change occurred.

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
