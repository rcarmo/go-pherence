# Nemotron ASR CPU trial on Sigma — 30 September 2026

Nemotron ASR has replaced Whisper in Sigma's live transcription service on port 8093. Complete jobs, including decode, queue and export publication, processed the 11-second JFK sample in 6.87 seconds and the 20-second podcast crop in 11.60 seconds. Both ran faster than real time.

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

Locale prompts are explicit: `auto=101`, `en=0`, `pt=13`, `fr=8`, `es=2`, `it=15`. Nil `PCMGenerationStream.PromptID` preserves existing callers' automatic prompt. Automatic English clips and the explicit English speaker profile have trained local evidence below; other explicit language paths have configuration/validation tests, not multilingual accuracy qualification. Automatic prompting reports transcript language `und`, because native inference does not return a detected locale.

The ASR stage writes a separate model-neutral `transcript` checkpoint directly. Verified decode and other completed stages survive retries. Cancellation or inference failure publishes no partial transcript and retries ASR from zero with fresh state. This implementation does not journal incremental ASR output or reuse Whisper window acknowledgements.

Word alignment is unavailable. The JSON explicitly declares `timing:"rnnt-emission-chunks-not-word-alignment-v1"`; `words` remains absent. WebVTT includes a note describing coarse RNN-T emission/input-chunk timing. Subwords crossing PCM calls are retained until whitespace, without inventing word boundaries. Speaker attribution preserves the conservative full-cue policy when word alignment is absent; some coarse cues may remain unlabelled. Existing Whisper transcript bytes stay unchanged when the optional timing field is absent.

ASR progress reports processed samples, not durable window acknowledgements. The HTTP progress snapshot accepts those counters during the `transcript` stage and the UI labels them as transcription.

## Validation

- Focused job/HTTP/model/server/web package tests passed.
- Full `go test -p 1 ./...`, `go vet -p 1 ./...`, and `go build -p 1 ./...` passed with NVIDIA disabled and CGO disabled.
- Pinned candidate metadata test passed without model payload loading or a listener.
- Regression tests cover fresh retry, cancellation atomicity, retained decode, language/prompt identity, malformed decisions, coarse-timing roundtrip, unchanged legacy JSON and ASR sample progress.
- `git diff --check` passed.

## Isolated service and authorised replacement

A CPU-only isolated service loaded the full 36-profile configuration in a network-disabled container with four CPU quota, 8 GiB memory and no swap. It loaded native Nemotron ASR, Community-1 and Nemotron diarization weights, without loading Whisper weights. JFK automatic transcription completed in 6.864 s (RTF 0.624), explicit-English transcription plus speakers in 8.892 s (RTF 0.808), and the podcast crop in 12.388 s (RTF 0.619). Podcast container peak memory was 7,115,608,064 bytes, including file-backed pages, below its cap. These results include decode, queue and export publication, and exclude service startup/model loading.

The helper initially expected `und` in the job's profile-language field, where the API reports requested `auto`; the actual transcript correctly reports `und`. That assertion was corrected. The first podcast poll hit the helper's eight-second HTTP timeout during inference and service shutdown cancelled that test job; a separate podcast-only run with a 35-second verifier timeout then completed. No failed output was accepted or replayed under an incompatible plan.

Rui authorised replacing the production service. The queue contained only an old terminal failed item. Before stopping the old unit, binaries/config were backed up under `deploy-backup-nemotron-asr-20260930T132502Z`. All seven prior manifest hashes matched after installation and before restart; no existing job data was removed. Profile IDs remain available, but new model/runtime stage identities do not accept old Whisper checkpoints as Nemotron output. Existing exports stay downloadable.

Installed binaries built from source commit `709676436fae504a22d3743dfc3a2e7dbd8a2601`:

- Server SHA256: `a39e5a5de32bf59e55019840dfe5e3926f5d49a96006e75da5b97f9ab7fb16ac`.
- Frontend SHA256: `247de771d6a2ba96bb802bfd2b113d6e6bb2d26df9b6155833b5d94207dbd663`.
- Installed config opts into `nemotron_asr`; its weights path names the pinned Nemotron ASR checkpoint. All 36 profiles use coarse timing and CPU execution. There are no Vulkan opt-ins.
- Metadata-only `--check` passed with `metadata_checked:true`, `model_loaded:false`, `listening:false`.

## Live complete-job measurements

The resident service was measured with one queued job at a time, `GOMAXPROCS=4`, systemd `CPUQuota=400%`, `MemoryMax=8G`, `MemorySwapMax=0`, and `PrivateDevices=yes`. The Qwen LAN slot was idle before native checks; its service remained available and unchanged. Gemma remained off as requested.

| Live case | Job ID | Audio | Complete job | RTF |
| --- | --- | ---: | ---: | ---: |
| Automatic JFK ASR | `e46a48ce4c003d621d756ffc0329f333` | 11 s | 6.870 s | 0.625 |
| English JFK ASR plus speakers | `3e0c7831d2b73debcce574b0da143deb` | 11 s | 8.942 s | 0.813 |
| Automatic podcast ASR | `06f0f3f603a84ed865d72dda278adf6a` | 20 s | 11.595 s | 0.580 |

Complete-job time is persisted `updated - created` before reconciliation: it includes upload creation, queue/decode, inference and export publication, excludes model loading, and does not include the verifier's extra cleanup wait. Each job completed in one attempt. JFK words matched the known sentence; podcast quality was not human-scored. Jobs have `decode`, `transcript`, `vtt` checkpoints and no `asr-windows` Whisper journal. The speaker job adds `diarization`, `speaker-transcript`, `speaker-vtt`.

All plain/speaker export downloads matched the API's size and SHA256. Successful reconciliation released source media, while transcript/VTT downloads remained available. SSE delivered ASR processed-sample progress and diarization sample counters. Transcript JSON and VTT explicitly state coarse emission-chunk timing and no word alignment. Speaker cues may remain unlabelled under conservative full-coverage attribution; this is not word-aligned speaker accuracy qualification.

Live cancellation/retry job `b74a71ac08e16b23bd175f615daf47e8` cancelled its first ASR attempt, retained source media and the verified decode checkpoint, and published no partial transcript. Explicit retry completed on attempt two with the same decode checkpoint key/blob and a downloadable transcript. ASR replayed from zero with fresh stream state.

Playwright checked Nemotron as the default speaker provider, four export controls, an actual browser VTT download, no collapsibles and no page errors. LAN `192.168.1.70:8093` reports 36 profiles. The service had no restarts, zero cgroup swap, and peak memory 6,916,440,064 bytes (about 6.44 GiB, including file-backed pages; not process RSS). Host available memory exceeded the 6 GiB guard. Qwen PID and restart count stayed unchanged.

## Remaining quality/performance scope

The replacement and faster-than-realtime complete-job target are verified for these short recordings on Sigma. Labelled multilingual/multi-speaker accuracy, word alignment, long-recording throughput, independent local PyTorch reference parity and contention performance remain unverified. No GPU execution or recovery was attempted.
