# Qwen3-TTS two-hour CPU retention

One approved 0.6B CustomVoice CPU checkpoint completed a two-hour retained-output
run. Thirteen rounds started about ten minutes apart; each round generated two
concurrent 32-frame “Hi”/Ryan/English/seed-42 requests. All 26 results matched
the pinned independent Rust codes and waveform fixture. Maximum waveform error
was `6.705522537231445e-7`, below the unchanged `1.6e-6` limit. Every earlier
output was rechecked while later rounds ran and at the end of the interval.
No results aliased. After releasing all 26 outputs, post-GC live heap was
534,832 bytes **below** the pre-run snapshot.

`--retain-2h` extends the existing `scripts/qwen3tts-admission` probe. It hashes
the approved model, codec, reference script, codes and waveform before loading.
It uses one process, one shared Talker/CodePredictor/Decoder12Hz, six Go workers,
13 rounds at ten-minute start offsets and a final check after at least two
hours. It does not start a second checkpoint process. The previous `--retain-3`
mode is unchanged and passed ordinary and race checks after the extension.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  go run ./scripts/qwen3tts-admission \
  /dev/shm/qwen3tts-0b6-customvoice --retain-2h
```

The run started at `2026-09-26T23:05:29Z`, completed at
`2026-09-27T01:06:05Z` and wrote `exit_code=0`. The final recheck occurred
at elapsed `2h0m32.56349851s`, with 26 results still held. The observed
maximum `/proc/self/status` `VmHWM` was 5,320,788 KiB. Retained post-GC
`HeapAlloc` rose from 4,117,399,176 bytes after round 0 to 4,135,593,360
bytes after round 12 while the 26 outputs accumulated. The final released
post-GC `HeapAlloc` was 4,115,384,208 bytes versus a pre-run
4,115,919,040 bytes. The high-water RSS did not keep rising with later rounds.
These are live heap and process-RSS observations, not a bound on all possible
prompts or concurrency.

The durable evidence is
`/workspace/tmp/qwen3tts-cap64-20260926/soak-2h/ordinary.log` and
`exit-status.txt`. It contains 13 round records, 26 result checks and 13
retained post-GC snapshots. An interrupted run would have remained partial;
the terminal file confirms this run finished. A separate three-round race
run passed but the full two-hour run was ordinary only. This does not qualify
a two-hour race workload, days-long retention, longer sentences, more than two
simultaneous 32-frame callers, cancellation during retained use or GPU
execution.
