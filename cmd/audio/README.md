# cmd/audio — speech & audio

| Command | Purpose |
|---|---|
| `whisper` | Whisper STT/translation (defaults to local large-v3-turbo weights; WAV direct and M4A/other inputs via ffmpeg; `-task`/`-language` prompt flags); supports the `WHISPER_ENC_H` EP-encoder + Go turbo-decoder hybrid |
| `diarize-vtt` | Speaker diarization → WebVTT |
| `speakercheck` | Speaker-embedding / verification check |
