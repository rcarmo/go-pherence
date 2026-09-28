# Qwen3-TTS 32-frame cancellation and 64-frame recovery

The approved 0.6B CustomVoice CPU runtime returned no partial output when a
seed-42 sentence request was cancelled after 32 complete acoustic frames. A
new request using the **same loaded models** then exhausted the 64-frame cap,
matched all 1,024 pinned independent Rust codec codes, and matched the
122,880-sample reference waveform with maximum absolute error
`1.2848759070038795e-6` below the unchanged `1.6e-6` gate. EOS remained
enabled and was not selected in the recovery request.

The opt-in released test `TestCappedSentenceSeed42CPUReleasedCancellationRecovery`
hashes the same six approved checkpoint/tokeniser assets and the sentence's
Rust oracle, observation, code fixture and external waveform as the [64-frame
qualification](qwen3-tts-cap64-parity-20260926.md). A test-owned context counts
the existing stage-boundary `Err()` calls. The 1,246th check is the next
continuation-frame boundary after the initial frame and 31 further acoustic
frames. It cancels there without a timer, global mutation or a forced length.
The test requires `context.Canceled` and a zero result, then regenerates the
same sentence on the same Talker, CodePredictor and Decoder12Hz objects and
checks every semantic code, acoustic code and waveform sample.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR=/workspace/tmp/qwen3tts-cap64-20260926/reference1 \
  go test ./model/qwen3tts \
  -run '^TestCappedSentenceSeed42CPUReleasedCancellationRecovery$' -count=1 -v
# Repeat with go test -race for the instrumented path.
```

| Gate | Result |
|---|---|
| Ordinary released test | Pass in 60.25 s; 32-frame cancellation, 64-frame recovery; peak process RSS 5,303,296 KiB. |
| Released `-race` test | Pass in 127.25 s; same 1,246th context check, exact codes and waveform error. |

The first race attempt ended without a test result during a process restart;
its log contains only `=== RUN`. The later successful race log is the gate.
Both runs used `GOMAXPROCS=6`, one checkpoint process and no GPU. The recovered
result was still retained at the final heap snapshot, so that snapshot is not
a post-release leak test. Previous short-request admission separately checked
output ownership and release.

This test exercises cooperative cancellation **between** frames. An individual
Talker layer, CodePredictor frame or waveform decode remains non-interruptible.
It supplies no hard cancellation-latency distribution, concurrent long-request
cancellation, hours-long retention, sentence-completion or intelligibility
claim. Evidence: `/workspace/tmp/qwen3tts-cap64-20260926/context-long-ordinary-rss.log`
and `context-long-race-retry.log`.
