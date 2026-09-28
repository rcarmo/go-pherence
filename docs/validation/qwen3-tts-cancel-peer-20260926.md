# Qwen3-TTS released cancellation with an active peer

A context-aware “Hi”/Ryan/English/seed-42 request can cancel after Prefill while
a simultaneous legacy “Hello world”/Ryan/English/seed-7 request runs on the
same approved 0.6B CustomVoice CPU model. The cancelled request returns
`context.Canceled` and no codes, hidden rows, logits or waveform. The peer
matches every pinned Rust codec code and its waveform; a later context-aware
Hi request on the same model also matches its independent fixture. Results are
owned and do not alias.

The opt-in `--cancel-peer` admission mode uses the existing hash-verified
16-frame fixtures and one shared Talker, CodePredictor and Decoder12Hz. A
test-owned context cancels at the second `Err()` check, reached just after
Talker Prefill. The check counts exactly two calls; no timer or wall-clock
threshold controls cancellation. Both requests start from one barrier. The
probe then checks peer parity, recovery, output independence, releases both
outputs, forces GC and measures live heap. It does not modify inference code.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  go run ./scripts/qwen3tts-admission \
  /dev/shm/qwen3tts-0b6-customvoice --cancel-peer
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  go run -race ./scripts/qwen3tts-admission \
  /dev/shm/qwen3tts-0b6-customvoice --cancel-peer
```

| Observation | Ordinary | Race |
|---|---:|---:|
| Two requests returned | 12.41 s wall | 23.50 s wall |
| Cancelled request context checks | 2 | 2 |
| Hello-world peer maximum waveform error | `7.152557373046875e-7` | same |
| Recovered Hi maximum waveform error | `4.917383193969727e-7` | same |
| Peak `/proc/self/status` `VmHWM` | 4,886,208 KiB | 14,248,540 KiB |
| Post-GC live-heap delta | −300,248 B | −263,272 B |

Ordinary RSS is the resource observation; race-detector shadow memory inflates
its RSS. Both runs used six Go workers and one checkpoint process. There was no
partial result, peer corruption, aliasing, reported data race or live-heap
growth above the existing 1 MiB guard. Wall times cover peer inference and
recovery, and do **not** measure a cancellation latency distribution.

The context-aware path checks between stages. Cancellation inside a Talker
layer, CodePredictor frame or waveform decoder cannot interrupt that operation.
The [two-frame context qualification](qwen3-tts-capped-cancellation-20260926.md)
and this released peer check do not qualify longer in-flight cancellation,
repeated cancellation over hours, a service workload or GPU execution. Raw logs
are `/workspace/tmp/qwen3tts-cap64-20260926/cancel-peer-ordinary.log` and
`cancel-peer-race.log`.
