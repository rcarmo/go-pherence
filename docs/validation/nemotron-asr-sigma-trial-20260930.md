# Nemotron ASR CPU trial on Sigma — 30 September 2026

Native Nemotron ASR transcribed the JFK and podcast clips faster than real time on Sigma. The experimental speech-job/server integration passes model-free tests, but has not replaced the deployed Whisper ASR service.

## Pinned assets

- Model: `nvidia/nemotron-3.5-asr-streaming-0.6b`, revision `ea30d66debe3740a08b573244286791d423d6b3e`.
- Safetensors: 2,552,062,944 bytes, SHA256 `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174`.
- Tokenizer SHA256: `3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246`.
- Config SHA256: `62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845`.
- Header inspection: 655 F32 tensors, 24 encoder layers, 128 mel bins. Asset hashes matched the pinned download and expected checkpoint identities. Assets are local and excluded from Git.

## CPU measurements

The trial ran in a network-disabled container with four CPU quota, `GOMAXPROCS=4`, 8 GiB memory and no container swap. No GPU device was mounted. The live Qwen slot was idle before the trial; Qwen and the transcription service were unchanged. The trial used Go's native streaming encoder and RNN-T decoder, automatic prompt 101, 80,000-sample PCM calls and a fresh stream for each request.

| Recording | Audio | Request | Real-time factor |
| --- | ---: | ---: | ---: |
| JFK, first stream | 11 s | 6.629 s | 0.603 |
| JFK, fresh repeated stream | 11 s | 6.282 s | 0.571 |
| Podcast, 300–320 s crop | 20 s | 11.252 s | 0.563 |

Request time includes frontend, encoder, RNN-T, returned decision collection and text decoding. It excludes model/tokenizer/WAV loading. Model/tokenizer load took 2.008 s. Process elapsed across all three requests was 26.296 s. Peak process RSS was 5,341,028 KiB (about 5.09 GiB); peak container memory was 5,479,555,072 bytes, including file-backed pages. Container swap stayed zero. Host swap pre-existed.

The two JFK requests produced identical tokens, encoder frame positions and text:

> And so my fellow Americans ask not what your country can do for you.  Ask what you can do for your country

NFKC, lowercase, punctuation-insensitive word scoring against the known 22-word sentence gave zero edits. This single-speaker recording does not establish general accuracy. The podcast text was saved but not scored against human annotations. The trial did not run an independent PyTorch oracle on Sigma.

The retained deployed Whisper JFK job spent about 28 s before ASR/transcript publication, including decode and queue overhead. It had the same 22 words. That timing and Nemotron's inference-only timing have different scopes and are not a controlled paired speedup measurement.

## Experimental integration

Top-level `nemotron_asr` opts the server into a separate Nemotron-only model loader and profile builder. It verifies the pinned metadata and token vocabulary, shares immutable model weights and creates fresh frontend/encoder/predictor state for each attempt. It does not load a second Whisper model or perform automatic fallback.

Locale prompts are explicit: `auto=101`, `en=0`, `pt=13`, `fr=8`, `es=2`, `it=15`. Nil `PCMGenerationStream.PromptID` preserves existing callers' automatic prompt. Only the automatic English clips above have trained local evidence; the explicit language paths have configuration/validation tests, not multilingual accuracy qualification. Automatic prompting reports transcript language `und`, because native inference does not return a detected locale.

The ASR stage writes a separate model-neutral `transcript` checkpoint directly. Verified decode and other completed stages survive retries. Cancellation or inference failure publishes no partial transcript and retries ASR from zero with fresh state. This implementation does not journal incremental ASR output or reuse Whisper window acknowledgements.

Word alignment is unavailable. The JSON explicitly declares `timing:"rnnt-emission-chunks-not-word-alignment-v1"`; `words` remains absent. WebVTT includes a note describing coarse RNN-T emission/input-chunk timing. Subwords crossing PCM calls are retained until whitespace, without inventing word boundaries. Speaker attribution preserves the conservative full-cue policy when word alignment is absent; some coarse cues may remain unlabelled. Existing Whisper transcript bytes stay unchanged when the optional timing field is absent.

ASR progress reports processed samples, not durable window acknowledgements. The HTTP progress snapshot accepts those counters during the `transcript` stage and the UI labels them as transcription.

## Validation and remaining deployment work

- Focused job/HTTP/model/server/web package tests passed.
- Full `go test -p 1 ./...`, `go vet -p 1 ./...`, and `go build -p 1 ./...` passed with NVIDIA disabled and CGO disabled.
- Pinned candidate metadata test passed without model payload loading or a listener.
- Regression tests cover fresh retry, cancellation atomicity, retained decode, language/prompt identity, malformed decisions, coarse-timing roundtrip, unchanged legacy JSON and ASR sample progress.
- `git diff --check` passed.

The live port-8093 service still uses Whisper ASR with Nemotron diarization. A bounded complete server/job smoke, resource validation with the larger ASR model, and idle-queue deployment are still required. Faster-than-realtime service completion, labelled multi-language accuracy, long-recording performance and independent reference parity remain unverified.
