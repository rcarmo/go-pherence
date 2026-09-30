# Nemotron speech-job CPU integration — 30 September 2026

The experimental Nemotron provider is deployed on Sigma's transcription service at port 8093. Real Whisper ASR plus CPU Nemotron completed the JFK smoke, with verified plain and speaker downloads. Community-1 remains an explicit speaker provider.

## Provider and checkpoint contract

- Released revision: `a435e9867d79e789e90053f9b6d6834053af564a`.
- Model SHA256: `c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6` (396,954,592 bytes; matched Hugging Face's linked ETag).
- Config: `ProfileSettings.nemotron`, with explicit experimental consent, revision, model path/hash and result-byte cap.
- Independent `speechjob-nemotron-diarization-cpu-stream-v1` identity. The provider writes its own canonical `provider:"nemotron"` document; it does not accept Community-1 documents.
- CPU stream factory shares immutable weights and resets frontend, pending PCM, speaker cache and request state on every attempt. It does not inherit device projectors or GPU towers.
- Input: mono canonical 16 kHz PCM, 160 samples through four hours; 80,000-sample batches. Result: at most 100,000 turns and 16 MiB. Native frame-count checks reject missing or extra output.
- Speaker policy: `nemotron-full-turns-maximum-positive-overlap-per-word-v1`. Full overlapping spans are preserved. Maximum positive overlap labels words; exact ties and no overlap remain unlabelled. Legacy cues require complete unambiguous coverage.
- Plain transcript/VTT stages precede diarization. Diarization failure/cancellation cannot remove those checkpoints. New-job UI selection uses `nem-*`; existing `asr-*` and `diar-*` IDs, profile settings and Community-1 algorithm version are retained. The rebuilt runtime SHA changes configuration/stage hashes. Old checkpoints are not rewritten or accepted under a mismatched plan; their exports remain accessible.

## Trained smoke

`TestNemotronTrainedCPUJobRetryChunkParity` ran inside a network-disabled container with four CPU quota, four Go threads, 4 GiB memory and no container swap. No GPU device was mounted. The live transcription and Gemma services were not changed.

The input was `testdata/jfk.wav` (11 seconds, 176,000 mono 16 kHz samples). The ASR transcript was a timing fixture, not a Whisper inference run.

The passing run checked:

1. Cancellation after the first 80,000 input samples reached diarization; the decode, transcript and plain VTT checkpoints remained intact and downloadable from the store.
2. Retry used fresh native state and reused all three prefix checkpoints without rerunning the transcript stage.
3. A separate native stream using 7,979-sample chunks produced identical turns to the job's 80,000-sample chunks.
4. All three fixture words received speaker labels. Their text/timestamps and transcript language were preserved. The experimental speaker VTT used `SPEAKER_00`.
5. No scratch files remained. Six final checkpoints were published.

Turns: `(0.31, 2.25, 0)`, `(3.29, 4.53, 0)`, `(5.40, 10.63, 0)`.

Test elapsed: 5.76 seconds, including cancellation, retry and the separately chunked run. This is not a single-pass throughput measurement. Host swap was already in use; only container swap was prohibited. No peak-RSS measurement was collected.

## Reference mismatch and limits

The first parity invocation incorrectly supplied `jfk_request_logits.f32.gz` to `TestReleasedPCMStreamingRequestPyTorchParity`. That checked-in fixture comes from the offline `nemotron_diarization_qkv_fixture.py` generator, with 1,101 rows. The low-latency stream emits 1,099 rows and differed substantially (`max_abs=11.824`, 8,790 values outside the test tolerance). The test correctly failed.

The subsequent passing smoke checks native chunk determinism and job integration only. An independently generated low-latency PyTorch reference is still required. Neither native chunk agreement nor the single-speaker JFK sample measures diarization quality, overlap quality or long-recording speed. The separate end-to-end check below covers real Whisper integration. Speaker output remains experimental; no local quality qualification was established.

## Full validation

Validation used Go 1.26.2, `CGO_ENABLED=0`, `GO_PHERENCE_DISABLE_NVIDIA=1` and bounded Go worker counts.

- `go test -p 2 ./...`: passed (133 packages with test results).
- The Makefile `test-cpu` command, with `-p 2` added: passed uncached (72 packages). `make` is absent on this host, so the recipe was run directly.
- `go vet -p 2 ./...` and `go build -p 2 ./...`: passed.
- Trimmed `speechjobserve` and `transcribe-web` builds: passed. Browser JavaScript bundled with Bun.
- Candidate config: 36 profiles, with the original 24 profile settings preserved, 65,063 bytes within the 64 KiB parser cap. Updated candidate runtime SHA and `speechjobserve --check` passed with `metadata_checked:true`, `model_loaded:false`, `listening:false`.
- `git diff --check`: passed.

The first full test run found an existing 1-ULP Whisper attention mismatch: the packed query path called the serial NT kernel while the per-head path selected blocked FMA. The packed path now selects NT arithmetic using the full head dimensions, preserving that choice for one-row tails and heads above the blocked cap. Original and added tail/cap/long-batch bit-exact regressions passed three times. Tolerances were not changed.

## Deployed end-to-end verification

Rui authorised replacing the production transcription service. The old queue had only one terminal failed entry. The old binaries/config/unit were backed up under `deploy-backup-nemotron-20260930T123227Z`; all six existing manifest hashes matched before the replacement started. Existing job data was not removed.

Clean committed server/frontend builds matched the earlier binary hashes. The installed config contains 36 profiles. All old Whisper Vulkan opt-ins were explicitly removed, including those copied to new Nemotron profiles: deployment is CPU-only. The original 24 profile IDs and non-device options remain; runtime and device changes intentionally create new plan identities. Existing exports remain accessible, but old checkpoints are not accepted under incompatible plans.

The service uses four threads and systemd `CPUQuota=400%`, `MemoryMax=8G`, `MemorySwapMax=0`, `PrivateDevices=yes`. The Vulkan environment is unset. No GPU device is available to the service. The co-resident Qwen LAN service was not stopped or changed.

- Installed server SHA256: `5d6ec053012e3461498aa62259b6f30e004c1cdcb8c3180f9194578f5b41ab1c`.
- Installed frontend SHA256: `94886c917a5d3523b7b1cde8a7d12698b0de92c85874644d11637e3a76a365fa`.
- Backend source commit: `8aa1b112fb8545e43799e797127584cc2dfd8a1d`.
- Port 8093 API on loopback and `192.168.1.70` reports 36 profiles. Nemotron is the UI default when speaker identification is enabled; Community-1 is selectable.

A network-disabled 4-CPU/8-GiB isolated service first completed actual Whisper→Nemotron JFK job `ac061df23d1ac7bf983407233cf12d01`: 30.15 seconds from creation through completion, one attempt, seven checkpoints. The verifier initially used the wrong profile/artifact response field names and had an undersized scratch-store reservation; those helper/config failures were corrected. Saved exports passed separate model-free HTTP-handler size/hash/timeline checks without rerunning the completed job.

Deployed job `79adb472ba91daf84e30e73ceb3c1bbf`, profile `nem-en-wav`, completed in one attempt. It produced the correct 11-second JFK sentence, 22 timed words and speaker labels on all 22 words. All four exports downloaded successfully with recorded hashes. Word text/timing, language and source timing were unchanged. Verified download workflow including reconciliation took 33.64 seconds; this includes polling and the three-second post-completion cleanup wait, so it is not a single-stage throughput measurement.

Authenticated SSE delivered live ASR-window and Nemotron sample counters. The helper stopped the subscription after observing completion through the job API; it did not record a terminal SSE event. Plain transcript/VTT downloads still succeeded after successful reconciliation released the source media. Playwright verified Nemotron default selection, Community-1 selection, four visible export controls, no collapsibles and no page errors.

The service had zero restarts, zero cgroup swap and peak cgroup memory `6865473536` bytes (about 6.39 GiB). This includes file-backed pages and is not process RSS. Host available memory was about 22 GiB after the smoke. Host swap was already present.

## Remaining qualification

Independent low-latency PyTorch parity and multi-speaker/long-recording quality remain unverified. Resource-heavy work is coordinated with the Qwen CPU evaluation session. TEF regeneration still requires the original recording.
