# Optional go-264 media backend

`media.NewGo264(Go264Config{})` implements the existing `media.Adapter` with the pure-Go frontend pinned at `v0.0.0-20260913160821-a109fec2a1db`. The entire go-264 project is MIT-licensed through its root `LICENSE`; upstream MIT notices and separate external dataset licences are retained. `speechjobserve` now requires an explicit `profile.media_backend`: `go264` is the shipped pure-Go profile choice and `ffmpeg` remains an explicit rollback. There is no native codec or subprocess inside the go-264 backend.

## Import and use

The adapter is a normal exported Go library API in `github.com/rcarmo/go-pherence/loader/audio/media`. It requires no special build tag, local replacement, FFmpeg installation or model weights. It builds with `CGO_ENABLED=0`.

```go
import (
    "context"
    "github.com/rcarmo/go-pherence/loader/audio/media"
)

func decode(ctx context.Context, input, output string) error {
    backend, err := media.NewGo264(media.Go264Config{})
    if err != nil {
        return err
    }
    var adapter media.Adapter = backend
    _, err = adapter.DecodeToFile(ctx, input, output)
    return err
}
```

Keep the source immutable and use a caller-private directory for both paths. Read the resulting canonical WAV through `media.OpenCanonicalPCM`; its positional float32 reads feed existing speech window APIs. `ExampleNewGo264` and an external-package decode/read test exercise only the public API.

For direct decoder/resampler access without go-pherence, import `github.com/rcarmo/go-264/audio` instead. The speech-job constructor `speechjob.NewFFmpegDecodeStage` still explicitly selects FFmpeg; this library merge does not silently change that job's backend or cache identity.

## Contract

- Same caller-owned immutable local regular-file/private-directory policy and supported filename/signature checks as the current adapter.
- PCM WAV and a narrow progressive AAC-LC MP4/M4A/MOV subset; no μ-law/float/extensible WAV, HE-AAC, encrypted/fragmented media, correlated PNS, fractional edits or unqualified tools. Unsupported data returns `ErrUnsupportedInput`.
- Canonical output is16kHz mono signed16-bit WAV. Actual `DecodeResult.Timeline.Samples` counts emitted frames; probe metadata is structural and does not certify payload validity.
- Input retained; mode0600 same-directory temporary output; strict frame/byte validation and atomic no-clobber publication. Cancellation and non-EOF decode failure remove the temporary artifact. Final context is checked before publication. A syscall already running is not forcibly interruptible.
- Publication has atomic visibility, not fsync/power-loss durability. The separate go-264 PCM segment store is available to a future job layer; it is not silently substituted here.
- `OpenCanonicalPCM` is unchanged. It owns/closes its descriptor, serialises positional float32 reads and retains the file. The new backend writes files conforming to that reader; it does not replace it with a stateful decoder wrapper.
- Limits are inclusive for the pure-Go output. FFmpeg's `-fs` ceiling has different truncation semantics and is still rejected at equality in that backend.

The provider's `ProbeMetadata` exposes pre-trim and edited source-rate extents plus priming, padding and leading silence. The adapter uses the pre-trim extent when validating the provider-owned MP4 timing plan, then persists the exact mapping in the canonical checkpoint. Output timeline remains canonical PCM. Unsupported source timeline forms fail instead of being guessed.

## Public-fixture qualification

MINDS-14 source revision `40ce77cb32a384e4d50a568e1ec39ac804019d33`, CC-BY-4.0 (PolyAI): two pt-PT and one fr-FR clips, hashes/references in `go264_public_test.go`. Originals are8kHz μ-law and are explicitly rejected. Offline FFmpeg creates supported PCM-WAV and48kHz AAC derivatives. Source files remain unchanged.

The initial provider pin produced a real pt-0 WAV WER regression:16→20 word edits. The resampler applied its downsampling transition cutoff while upsampling. Provider `ec62711` (formerly `faa324c` before the owner-requested identity correction) corrects interpolation to preserve input samples at integer phases; independent signal tests pass. Frozen-pin rerun of all six media pairs yields identical Tiny token/segment/window results and zero added word edits. PT/FR baseline recognition is poor; this establishes paired frontend non-regression on those clips, not production ASR quality.

Public pyannote tutorial30s two-speaker sample/RTTM revision `b749285c5cdd4636b2edc7f766f1352c8dde9369` (repository MIT notice, CNRS): saved offline Community-1 reference comparison with overlap included and explicit0–30s evaluation region. WAV DER/boundaries identical. AAC DER delta+0.08514 percentage points at0.25s collar, effectively zero at0collar; nearest-boundary Hausdorff difference16.875ms. Gate was≤0.5percentage-point DER; boundary check≤250ms. The reference was cached CPU PyAnnote, not a serving or Go runtime dependency. This does not qualify go-pherence's separate Go diarization implementation or meeting corpora. The DER run overlapped six seconds of another session's encoder setup; saved functional predictions/scores are retained, but its timing is not an isolated performance result. The boundary-distance threshold was a post-run diagnostic guard, not a pre-registered tuning target.

## Build and release state

Focused adapter tests/vet, audio CLI build and Linuxarm64 media build pass. Whole-repository build has the same pre-existing SpacemiT/diffusion command failures as base `f141e63`; the optional backend does not repair them. Neural tests require explicit opt-in and host admission.

The adapter is integrated into the speech development branch with owner approval, as an explicitly selected library backend. The earlier scoped implementation and final artifact/licence reviews passed. Author/committer metadata was corrected to `Rui Carmo <rcarmo@users.noreply.github.com>` before integration; reviewed implementation `09d2db2` maps to `b38803f`, with unchanged source. The merge preserves the durable speech-job checkpoint and existing FFmpeg defaults. This is not a service deployment or a default-backend switch.

Provider `a47077e` is published. It adds the root MIT licence and notice/documentation cleanup on the identity-corrected history. Go/assembly source is unchanged from the previously qualified runtime; saved neural results are historical evidence, not a new model run. No `replace` directive is committed. A fresh public Go-client download with `GOSUMDB=sum.golang.org` verifies the new exact pin and ZIP checksum `h1:an+NkPCIdNsFvqamM7cV2NIMsSrwlFQ5NQL61N0FnL8=`; go.mod checksum remains `h1:DfT0hsjYj65x9Y1WRU9gNO/yvZZlvCorI8t4gQxQMX4=`.

The earlier sealed handoff, failed directory-entry ZIP and old hashes remain historical artifacts. They must not be edited to appear to contain the new pin. Use the refreshed proxy/bundles and recorded revision map for this checkout.

The provider has no external Go requirements. The consumer already requires purego (Apache-2.0), x/sys (BSD-3-Clause), yaml.v3 (MIT and Apache-2.0 by file), and yaml's check.v1 test dependency (BSD-2-Clause). Those inherited dependencies are not new MIT dependencies and the entire consumer is not MIT-only. Rui's project-wide MIT confirmation and the provider root licence also cover the video code, CLI and generator helpers; the former audio-only scope has been removed.

General Go Community-1 DER acceptance is unqualified; later one-sample results do not remove the retained strict intermediate/tie-handling failures. Further model runs require fresh explicit admission even when the host is idle.

Commands:

```sh
make speech-go264-check
# Requires GO_PHERENCE_MINDS_FIXTURE_DIR and optional retained output path:
make speech-go264-public-check
# Explicit admitted CPU model run; no GPU:
make speech-go264-paired-check
```

Provider `a109fec` is published. It adds `audio.ProbeMetadata`, which fixes the consumer's former misuse of the edited `Probe` frame count as the pre-trim input to `Track.TimingPlan`. The public three-clip WAV/AAC media matrix and speech-job WAV/AAC decode checks now pass with the new pin. Promoting the server profile changes only new explicitly configured profile identities; existing deployed services were not changed or restarted.
