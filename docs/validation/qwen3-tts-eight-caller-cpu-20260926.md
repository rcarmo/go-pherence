# Qwen3-TTS eight-caller CPU admission

Eight simultaneous short requests passed on one approved 0.6B CustomVoice
checkpoint with six Go workers. Four callers used the pinned “Hi”/Ryan/English/
seed-42 16-frame fixture, and four used the pinned “Hello world”/Ryan/English/
seed-7 16-frame fixture. All semantic and acoustic codes matched their
independent Rust references. All eight waveforms passed their existing
thresholds: maximum errors `4.917383193969727e-7` for “Hi” and
`7.152557373046875e-7` for “Hello world”.

The opt-in `--eight-mixed` mode extends the [four-caller probe](qwen3-tts-four-caller-cpu-20260926.md)
without changing the inference runtime. It hashes the same approved checkpoint,
reference scripts, codes and waveforms before loading. A cap-65 request returns
an empty result and error before the batch. Eight goroutines start at a common
barrier and share immutable Talker, CodePredictor and Decoder12Hz weights.
Every result has separate semantic, acoustic and waveform storage; mutation of
request 0's waveform leaves the other seven reference comparisons unchanged.
All outputs are released before a post-GC heap measurement.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  go run ./scripts/qwen3tts-admission \
  /dev/shm/qwen3tts-0b6-customvoice --eight-mixed
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  go run -race ./scripts/qwen3tts-admission \
  /dev/shm/qwen3tts-0b6-customvoice --eight-mixed
```

| Result | Ordinary | Race |
|---|---:|---:|
| Eight returned | 39.28 s wall | 72.41 s wall |
| Peak `/proc/self/status` `VmHWM` | 6,746,052 KiB | 20,177,816 KiB |
| Post-GC live-heap delta after output release | −297,016 B | −297,080 B |

The ordinary peak RSS is the resource observation. Race instrumentation adds
shadow memory, so its RSS is not comparable. Both runs passed without aliasing,
a reported data race, partial results or live-heap growth above the existing
1 MiB guard. The check runs one short mixed batch; it does not qualify nine or
more callers, long prompts, hours-long retention, service throughput, or
cancellation while other real requests are active. GPU execution remains held.

Raw logs: `/workspace/tmp/qwen3tts-cap64-20260926/eight-mixed-ordinary.log`
and `eight-mixed-race.log`. The earlier [64-frame sentence gate](qwen3-tts-cap64-parity-20260926.md)
uses one request and must not be extrapolated to eight 64-frame callers.
