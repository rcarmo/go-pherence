# Nemotron speech-job CPU integration — 30 September 2026

The experimental Nemotron provider passed a bounded trained CPU job smoke. It is not deployed. Whisper remains the ASR engine; Community-1 remains an explicit speaker provider.

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

The subsequent passing smoke checks native chunk determinism and job integration only. An independently generated low-latency PyTorch reference is still required. Neither that native agreement nor the single-speaker JFK sample measures diarization quality, overlap quality, long-recording speed or trained end-to-end Whisper integration. Speaker output remains experimental and the HTTP view remains `qualified:false`.

## Full validation

Validation used Go 1.26.2, `CGO_ENABLED=0`, `GO_PHERENCE_DISABLE_NVIDIA=1` and bounded Go worker counts.

- `go test -p 2 ./...`: passed (133 packages with test results).
- The Makefile `test-cpu` command, with `-p 2` added: passed uncached (72 packages). `make` is absent on this host, so the recipe was run directly.
- `go vet -p 2 ./...` and `go build -p 2 ./...`: passed.
- Trimmed `speechjobserve` and `transcribe-web` builds: passed. Browser JavaScript bundled with Bun.
- Candidate config: 36 profiles, with the original 24 profile settings preserved, 65,063 bytes within the 64 KiB parser cap. Updated candidate runtime SHA and `speechjobserve --check` passed with `metadata_checked:true`, `model_loaded:false`, `listening:false`.
- `git diff --check`: passed.

The first full test run found an existing 1-ULP Whisper attention mismatch: the packed query path called the serial NT kernel while the per-head path selected blocked FMA. The packed path now selects NT arithmetic using the full head dimensions, preserving that choice for one-row tails and heads above the blocked cap. Original and added tail/cap/long-batch bit-exact regressions passed three times. Tolerances were not changed.

## Remaining release work

Commit/push and idle-queue API/browser deployment verification remain. Independent low-latency PyTorch parity and multi-speaker/long-recording quality remain unverified. Resource-heavy work is coordinated with the Qwen CPU evaluation session. TEF regeneration still requires the original recording.
