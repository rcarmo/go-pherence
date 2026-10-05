# transcribe-web redeploy from f3142b3e — 2 October 2026

`speechjobserve` on Sigma port 8093 now runs source `f3142b3e`, with server SHA256 `91d2080447f8150beacad780f40f01ed4a5c1eb3b849d445136732c7a4c25056`. The rebuilt `transcribe-web` front end is byte-identical to the deployed one (`247de771…`), so it was not replaced.

## What changed for the app

Nothing user-visible. The app's ASR engine is CPU Nemotron (top-level weights), and the unit's `nemotron-cpu.conf` drop-in keeps it off the GPU: `PrivateDevices=yes`, `CPUQuota=400%`, `MemoryMax=8G`, no swap. Since the previous server (`c9569c86`), the server code gained only an opt-in Nemotron Vulkan projection, which is not configured. The Whisper speed-ups ([final comparison](whisper-final-comparison-20261002.md)) are library work on the Vulkan Q5 path, which the app does not load.

## Procedure and checks

- Package tests: `cmd/audio/transcribe-web`, `cmd/audio/speechjobserve`, `runtime/speechjob` and `runtime/speechjob/httpapi`. The full `go test ./...`, race and cross-builds ran before the source commits.
- `speechjobserve --check` on the candidate config passed (`metadata_checked`, no model load). The config differs from the deployed one only in `runtime_sha256`.
- The backup is `deploy-backup-whisper-f3142b3e-20261002T103909Z`: previous config, server binary and the SHA256s of 20 job manifests.
- Install was by atomic rename; then the user unit restarted. Afterwards the unit is active with 0 restarts and 36 profiles, and the sandbox is unchanged. All 20 earlier manifests are byte-unchanged.
- Smoke test: JFK on `nem-en-wav` completed in 7.0 s wall. All four exports (transcript JSON/VTT, speaker JSON/VTT) were verified by size, SHA256 and ETag. The ASR VTT is byte-identical to the previous binary's JFK job (`50d31963`); the transcript JSON differs only in the language tag (`en` against that job's `und`).

## Existing gap found (not a regression)

Since Nemotron replaced Whisper ASR (30 September), JFK speaker exports label **0 words** (`labelled_words: 0`, earlier jobs `3e0c7831` and `b3b0ce6f`). The last Whisper-ASR job (`79adb472`) labelled 22 of 22. Nemotron's coarse word timing does not overlap the diarization turns under the maximum-positive-overlap policy. The speaker exports are therefore currently empty of labels.
