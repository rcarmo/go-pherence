# Source-timing output provenance — 12 September 2026

This checkpoint carries the optional validated decode `SourceTiming` mapping into final transcript and experimental diarization artifacts without changing canonical timestamp semantics.

## Contract

- `Transcript` schema 2 requires `source_timing` and retains cues as integer mono-16-kHz canonical PCM samples.
- `DiarizationDocument` schema 2 requires the same mapping and retains raw canonical reconstruction-frame-centre turns.
- Zero-valued mappings remain valid for legacy/plain canonical WAV inputs. Nonzero mappings use the same strict validation as the durable `gptm` WAV chunk.
- `NewTranscriptStage` and the Community-1 stage copy mapping metadata from the verified decode checkpoint; they do not infer it from duration or PCM length.
- Speaker output rejects transcript/diarization inputs whose source mappings differ.
- Stage identities for transcript, VTT, diarization, speaker transcript and speaker VTT are revised, preventing silent reuse of schema-1 artifacts.
- `Exact=false` remains explicit. No code converts canonical cue/turn times to source PTS or claims exact edit-list alignment.

## Verification

- `go test ./runtime/speechjob ./runtime/speechjob/httpapi ./cmd/audio/speechjob` passed.
- Thirty shuffled `runtime/speechjob` repetitions passed.
- Three explicit real-FFmpeg WAV/AAC job repetitions passed with deterministic decode checkpoints.
- Affected `go vet`, Linux/ARM64 and Windows/AMD64 media/job test-binary cross-builds, and `git diff --check` passed.
- Focused `cmd/audio/speechjobserve` tests passed separately; the first broad combined command exceeded its outer harness timeout after preceding packages passed, with no reported test failure.

No model, GPU, trained inference, private audio, service, deployment, media backend default, gate, tolerance or dependency pin changed. Broad real MOV edit-list/priming fixtures remain open.
