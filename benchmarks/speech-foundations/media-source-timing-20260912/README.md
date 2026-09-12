# Media source-timing boundary — 12 September 2026

This checkpoint makes source timing metadata explicit at the `loader/audio/media` adapter boundary without changing the canonical WAV checkpoint format or the default FFmpeg backend.

## Contract

- `Timeline` remains the authoritative decoded/canonical sample count.
- `SourceTiming` separately reports selected-stream start, duration, source rate, edit presence, priming, end padding and leading silence.
- FFmpeg probes `start_time`, rejects negative/nonfinite values, preserves source start/duration and marks MOV/MP4 mapping `Exact=false` because the narrow ffprobe contract does not prove edit-list and encoder-delay decomposition.
- WAV with zero source start is exact.
- The optional go-264 adapter maps its public checked MP4 `TimingPlan` and decoder metadata into exact source-rate fields.
- The speech-job decode boundary rejects negative timing/count metadata but intentionally keeps its canonical PCM payload unchanged. Transcript and diarization times therefore remain PCM-relative until a separate persisted mapping schema is implemented.

## Verification

- `go test ./loader/audio/media ./runtime/speechjob` passed.
- Ten shuffled repetitions of media, speech-job and server packages passed before the final timing assertions.
- Three explicit synthetic FFmpeg integration repetitions passed for 44.1/48 kHz WAV and AAC-generated M4A.
- Affected `go vet` and `git diff --check` pass.

No production media, private audio, model, GPU, service, deployment, pin or default was changed. Broader real MOV edit-list/priming fixtures and durable source-time publication remain open.
