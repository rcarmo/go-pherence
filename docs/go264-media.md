# Optional go-264 media backend

`media.NewGo264(Go264Config{})` implements the existing `media.Adapter` with the MIT pure-Go frontend pinned at `v0.0.0-20260912083556-faa324c3f8c4`. `NewFFmpeg` and existing serving defaults are unchanged. There is no native codec or subprocess inside this optional backend.

## Contract

- Same caller-owned immutable local regular-file/private-directory policy and supported filename/signature checks as the current adapter.
- PCM WAV and a narrow progressive AAC-LC MP4/M4A/MOV subset; no μ-law/float/extensible WAV, HE-AAC, encrypted/fragmented media, correlated PNS, fractional edits or unqualified tools. Unsupported data returns `ErrUnsupportedInput`.
- Canonical output is16kHz mono signed16-bit WAV. Actual `DecodeResult.Timeline.Samples` counts emitted frames; probe metadata is structural and does not certify payload validity.
- Input retained; mode0600 same-directory temporary output; strict frame/byte validation and atomic no-clobber publication. Cancellation and non-EOF decode failure remove the temporary artifact. Final context is checked before publication. A syscall already running is not forcibly interruptible.
- Publication has atomic visibility, not fsync/power-loss durability. The separate go-264 PCM segment store is available to a future job layer; it is not silently substituted here.
- `OpenCanonicalPCM` is unchanged. It owns/closes its descriptor, serialises positional float32 reads and retains the file. The new backend writes files conforming to that reader; it does not replace it with a stateful decoder wrapper.
- Limits are inclusive for the pure-Go output. FFmpeg's `-fs` ceiling has different truncation semantics and is still rejected at equality in that backend.

The provider's exact seek/priming/padding metadata is not exposed as original container PTS in this Adapter interface. Output timeline remains canonical PCM. Unsupported source timeline forms fail instead of being guessed.

## Public-fixture qualification

MINDS-14 source revision `40ce77cb32a384e4d50a568e1ec39ac804019d33`, CC-BY-4.0 (PolyAI): two pt-PT and one fr-FR clips, hashes/references in `go264_public_test.go`. Originals are8kHz μ-law and are explicitly rejected. Offline FFmpeg creates supported PCM-WAV and48kHz AAC derivatives. Source files remain unchanged.

The initial provider pin produced a real pt-0 WAV WER regression:16→20 word edits. The resampler applied its downsampling transition cutoff while upsampling. Provider `faa324c` corrects interpolation to preserve input samples at integer phases; independent signal tests pass. Frozen-pin rerun of all six media pairs yields identical Tiny token/segment/window results and zero added word edits. PT/FR baseline recognition is poor; this establishes paired frontend non-regression on those clips, not production ASR quality.

Public pyannote tutorial30s two-speaker sample/RTTM revision `b749285c5cdd4636b2edc7f766f1352c8dde9369` (repository MIT notice, CNRS): saved offline Community-1 reference comparison with overlap included and explicit0–30s evaluation region. WAV DER/boundaries identical. AAC DER delta+0.08514 percentage points at0.25s collar, effectively zero at0collar; nearest-boundary Hausdorff difference16.875ms. Gate was≤0.5percentage-point DER; boundary check≤250ms. The reference was cached CPU PyAnnote, not a serving or Go runtime dependency. This does not qualify go-pherence's separate Go diarization implementation or meeting corpora.

## Build and release state

Focused adapter tests/vet, audio CLI build and Linuxarm64 media build pass. Whole-repository build has the same pre-existing SpacemiT/diffusion command failures as base `f141e63`; the optional backend does not repair them. Neural tests require explicit opt-in and host admission.

The provider commit is local and has not been pushed. A temporary file-based Go module proxy supplied its exact Git-archived pseudo-version for verification. No `replace` directive is committed. Fresh remote consumers cannot fetch this dependency until that provider revision is published or the verified proxy bundle is supplied. Do not merge as a generally fetchable release without resolving publication.

Commands:

```sh
make speech-go264-check
# Requires GO_PHERENCE_MINDS_FIXTURE_DIR and optional retained output path:
make speech-go264-public-check
# Explicit admitted CPU model run; no GPU:
make speech-go264-paired-check
```

No service/backend switch, model download, Git push or release tag was performed.
