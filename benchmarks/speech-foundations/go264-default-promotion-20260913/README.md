# Pure-Go speech-job media promotion — 13 September 2026

The speech-job server now requires an explicit media backend and the shipped examples select the pinned pure-Go go-264 provider. FFmpeg remains an explicit rollback profile. No service was deployed or restarted.

## Provider correction

The initial audit at provider `48ff0ca` exposed a real consumer bug on all three public 48 kHz AAC-LC M4A fixtures. `audio.Probe` returns the edited source-rate extent, but the adapter passed that value to `Track.TimingPlan`, whose input is the pre-trim decoded extent. The result was `audio: malformed: decoded frames too short for timing`.

Provider `a109fec2a1dbb70646c0ac649d67a83a1fe0d6af`, published as `v0.0.0-20260913160821-a109fec2a1db`, adds `audio.ProbeMetadata`. It exposes both extents plus priming, padding and leading silence. The adapter now uses that provider-owned metadata and no longer reconstructs the timing plan from the edited count.

The failed pre-fix run is retained in `provider-public.log` with exit status 1. All PCM-WAV arms passed; all AAC arms failed before the correction.

## Server contract

- `profile.media_backend` is mandatory.
- `go264` selects the pinned pure-Go WAV/progressive-AAC-LC backend and rejects FFmpeg assets.
- `ffmpeg` selects the existing rollback constructor and requires both absolute executable paths and SHA-256 values.
- Decode stage versions include the backend and complete configuration. Old FFmpeg checkpoints cannot be reused as go-264 checkpoints.
- Both backends keep the existing private scratch, quota, cancellation, canonical-WAV, source-timing and durable-checkpoint contracts.

## Verification

With `GOMAXPROCS=2`, `CGO_ENABLED=0` and NVIDIA disabled:

- full affected adapter/job/HTTP/CLI/server package tests pass;
- affected vet passes;
- pure-Go decode identity, provider pin, canonical WAV, resume and cancellation pass ten repetitions;
- pure-Go synthetic AAC speech-job decode passes three repetitions with exactly 16,000 canonical frames and exact source metadata: 48 kHz, 1,024 priming frames and 128 padding frames;
- explicit FFmpeg WAV/AAC rollback tests pass three repetitions;
- generated pure-Go server profile runs WAV through `decode → asr-windows → transcript → vtt` without FFmpeg;
- the public MINDS matrix passes all six media arms: three PCM-WAV and three 48 kHz AAC-LC M4A variants. AAC SNR versus FFmpeg is 82.68–86.17 dB and maximum absolute difference is at most `0.000152587890625`;
- JSON examples parse and select `go264` explicitly;
- `git diff --check` passes.

Provider support stays narrow and fail-closed for unqualified AAC tools and container forms. Broader codec conformance and native ARM64 performance remain separate provider qualification work unless the provider owner reports otherwise.
