# Media source-timing boundary — 12 September 2026

This checkpoint makes source timing metadata explicit at the `loader/audio/media` adapter boundary and persists it through the durable canonical WAV checkpoint without changing the default FFmpeg backend.

## Contract

- `Timeline` remains the authoritative decoded/canonical sample count.
- `SourceTiming` separately reports selected-stream start, duration, source rate, edit presence, priming, end padding and leading silence.
- FFmpeg probes `start_time`, rejects negative/nonfinite values, preserves source start/duration and marks MOV/MP4 mapping `Exact=false` because the narrow ffprobe contract does not prove edit-list and encoder-delay decomposition.
- WAV with zero source start is exact.
- The optional go-264 adapter maps its public checked MP4 `TimingPlan` and decoder metadata into exact source-rate fields.
- The speech-job decode boundary writes a deterministic fixed-size private `gptm` RIFF chunk between `fmt ` and `data` whenever `SourceTiming` is present. Legacy/no-mapping checkpoints retain the exact 44-byte layout.
- `OpenCanonicalPCM.SourceTiming` round-trips that mapping. Duplicate, partial, reserved-bit, negative and overflowing values fail closed. Unknown legal RIFF chunks remain accepted and data offsets are scanned rather than assumed.
- The decode stage identity is revised so an old fixed-header decode checkpoint cannot be silently reused as the new source-timing contract.
- Canonical s16 PCM and `Timeline.Samples` remain authoritative. Transcript and diarization times remain PCM-relative; callers may map them to source time only according to persisted metadata and must not claim exact alignment when `Exact=false`.

## Verification

- `go test ./loader/audio/media ./runtime/speechjob` passed.
- Thirty shuffled repetitions of both affected packages passed with the final round-trip, legacy-layout, offset, malformed-chunk, deterministic-byte and retry/reopen assertions.
- Affected `go vet`, Linux/ARM64 and Windows/AMD64 test-binary cross-builds, server/CLI/HTTP regression packages, and `git diff --check` pass.
- Three explicit synthetic FFmpeg integration repetitions passed. Generated WAV retained exact PCM and exact timing; AAC-generated M4A retained deterministic PCM/timing checkpoints with mapping explicitly non-exact.

No production media, private audio, model, GPU, service, deployment, pin or default was changed. Broader real MOV edit-list/priming fixtures remain open.
