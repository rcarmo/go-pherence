# Integrated Whisper and Community-1

The speech port is being implemented in go-pherence on `feat/speech-simd-vulkan`, starting from `d08ce322292847a7978dcace1cc73a50d0bf8450`. Reference: Rui's **message:1671 / attachment:167**, implementation-sidebar request of 11 September 2026, and the subsequent clarification to use FFmpeg temporarily. Go owns inference; high-performance Go/Plan 9 assembly SIMD and Vulkan are first-class backends.

## Current implementation

- `loader/audio/media`: a separate, temporary FFmpeg/ffprobe adapter. Probe and decode accept context cancellation and return format/sample metadata. Conversion produces mono 16 kHz signed 16-bit RIFF/WAV without deleting the input or replacing an existing destination.
- `loader/audio.WhisperLogMel`: checked exact-contract 80/128-bin features, FFT400 geometry, periodic Hann, centred reflect padding, Slaney filters, final-frame removal and Whisper normalisation. The current DFT uses the existing checked SIMD `Ddot`; the mixed-radix Plan 9 FFT optimisation has not been implemented.
- `models/whisper.MelFlatFromSamplesChecked`: model-configured entry point to those features. Existing inference/CLI defaults remain unchanged pending real-checkpoint multilingual/timestamp qualification. The legacy 512-point/GPU mel path remains a separate implementation and is not the exact oracle.
- Four new synthetic 128-band fixtures generated independently with checksum-pinned Transformers 4.57.1 numerical functions; absolute tolerance `1e-5`. Existing 80-band reference still passes. No model weights are used by these tests.

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

## Exact frontend contract

`WhisperLogMel(samples, bands)` accepts 160..480000 mono 16 kHz float samples, `bands=80` or `128`, and rejects non-finite values. It does not resample or right-pad. It returns channel-major `[bands, floor(samples/160)]` output. Window planning, padding to model geometry and mapping words to original sample indices remain caller responsibilities. Do not pass a whole recording to one feature call.

The `WhisperLogMel80` compatibility API delegates to the new implementation; invalid inputs return no features. MOSS uses the 80-bin path on its already-padded windows. No model defaults or weight formats have been silently changed.

`loader/audio/testdata/whisper_logmel128_transformers_4_57_1.json` contains synthetic broadband, impulse/odd-boundary, short-reflect and silence cases. `scripts/whisper_logmel_reference.py` uses NumPy and only six checksum-pinned upstream numerical definitions. It does not import Transformers/PyTorch or load a checkpoint. See fixture metadata for source URL/hash and NumPy version. This reference is the NumPy Transformers frontend; exact Torch/whisper.cpp numerical differences still need separate real-model qualification.

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
# Short synthetic media only; requires ffmpeg and ffprobe installed.
GOMAXPROCS=2 CGO_ENABLED=0 make speech-media-integration
```

Set `TMPDIR` and `GOTMPDIR` to an existing writable directory if `/workspace/tmp` is unavailable. Use Go 1.26.2 as required by `go.mod`.

This checkpoint passed the focused make targets, affected-package vet, all audio-command builds and additional existing CPU convolution/LayerNorm/attention reference tests. The JSON record counts 54 passing test/subtest events and one intentionally skipped opt-in FFmpeg test; that FFmpeg test then passed in its separate enabled target. This is not a count of 54 top-level tests. No models, GPU calls or service changes occurred. See [verification evidence](../benchmarks/speech-foundations/verification.json).

`go build ./...` failed in the unrelated `backends/spacemit/aicpu/aipool`, `cmd/diffusiongemmainspect` and `cmd/diffusiongemmaserve` packages. A clean archive of untouched `d08ce322` produced identical errors; [baseline error listing](../benchmarks/speech-foundations/full-build-known-errors.txt) is retained. Those source files were not modified. Full model/backend suites and race tests have not been run. A delegated read-only review timed out without findings; parent source review and the listed tests were completed. The entire speech port and speed targets are not complete.
