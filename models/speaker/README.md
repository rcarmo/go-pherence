# models/speaker

Speaker embedding, verification, and diarization (SpeechBrain ECAPA-TDNN).

| Area | Files |
|---|---|
| Feature frontend | `fbank.go` (SpeechBrain filterbank), `speechbrain_forward.go` (conv1d/TDNN) |
| Embedding model | `ecapa.go`, `embed.go`, `speechbrain_ecapa.go`, `speechbrain_embed.go` |
| Diarization | `diarize.go`, `align.go`, `smooth.go` |
| Config / load | `config.go`, `load.go` |

The fbank/conv frontend here is SpeechBrain-specific (Kaldi-style fbank), distinct
from Whisper's log-mel — they are different feature pipelines, not duplicates.
