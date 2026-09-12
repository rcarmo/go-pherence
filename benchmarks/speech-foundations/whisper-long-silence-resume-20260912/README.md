# Whisper long exact-zero resume — 12 September 2026

This model-free checkpoint verifies the existing opt-in digital-silence shortcut through the durable speech-job journal over 131 windows.

## Contract

- `SkipDigitalSilence=true` remains explicit and defaults to false.
- Only exact PCM zero (including signed zero and padded tail) skips frontend/model execution. Nonzero subnormals and nonfinite samples remain rejected or processed by existing validation.
- Every skipped window still emits and durably acknowledges one canonical empty `windowRecord`; no timeline interval disappears.
- An injected first acknowledgement failure leaves the decode checkpoint and journal state recoverable. Reopen resumes at the first unacknowledged window and produces all 131 ordered records.
- This is not VAD, an energy threshold, or general no-speech quality evidence.

## Verification

- Focused long-silence failure/reopen/resume and existing >100-window tests passed.
- The full `runtime/speechjob` package passes with this coverage.
- No trained model or GPU runs; the toy model is deliberately invalidated by the exact-zero bypass after normal construction/preflight.
