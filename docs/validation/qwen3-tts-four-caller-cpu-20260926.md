# Qwen3-TTS four-caller CPU admission

Four simultaneous capped requests passed with one shared approved 0.6B
CustomVoice checkpoint. Two callers used the pinned “Hi”/Ryan/English/seed-42
16-frame fixture, and two used the pinned “Hello world”/Ryan/English/seed-7
16-frame fixture. All semantic and acoustic codes matched each caller's
independent Rust reference; waveform errors stayed below the fixtures'
unchanged thresholds. This is a short mixed-input concurrency check, not an
admission for longer prompts or more than four callers.

The opt-in `--four-mixed` mode in `scripts/qwen3tts-admission` reuses the
existing checkpoint, reference-script, code and waveform hashes. It constructs
one Talker, CodePredictor, Decoder12Hz and tokenizer, releases all four callers
at one start barrier, then checks every result. A cap-65 request is rejected
without partial output before the concurrent batch. Outputs are disjoint;
mutating request 0's waveform leaves the other three reference comparisons
unchanged. The results are released before a post-GC heap measurement.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  go run ./scripts/qwen3tts-admission \
  /dev/shm/qwen3tts-0b6-customvoice --four-mixed
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  go run -race ./scripts/qwen3tts-admission \
  /dev/shm/qwen3tts-0b6-customvoice --four-mixed
```

| Result | Ordinary | Race |
|---|---:|---:|
| Four callers returned | 19.95 s wall | 38.73 s wall |
| “Hi” requests 0 and 2, maximum waveform error each | `4.917383193969727e-7` | same |
| “Hello world” requests 1 and 3, maximum waveform error each | `7.152557373046875e-7` | same |
| Peak `/proc/self/status` `VmHWM` | 5,389,880 KiB | 16,138,360 KiB |
| Post-GC live-heap delta after output release | −298,880 B | −293,880 B |

Ordinary peak RSS is the admission observation; the race detector's shadow
memory makes its RSS unsuitable for that comparison. Both runs used six Go
workers and one checkpoint process. No output alias, data race, panic or
retained-heap growth above the 1 MiB guard appeared. The probe measures only
one batch plus post-release GC, not hours-long retention or a throughput
service. Cancellation, broader prompt coverage, five or more real callers,
foreign native execution and GPU work are still open.

Raw logs: `/workspace/tmp/qwen3tts-cap64-20260926/four-mixed-ordinary-final.log`
and `four-mixed-race-final.log`. Earlier two-request and retained-six-output
checks remain in [the sixteen-frame record](qwen3-tts-sixteen-frame-20260923.md).
