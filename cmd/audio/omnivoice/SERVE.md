# Persistent local speech worker

`-mode serve` runs a sequential NDJSON worker over stdin/stdout. It loads a fixed
model, cached reference, tokenizer, codec and reusable inference buffers once.
There is no HTTP listener, authentication service or background installation.

```sh
bin/omnivoice -mode serve -model "$MODEL" \
  -reference-tokens voice-codes.json -output-dir /private/existing/directory \
  -frames 75 -steps 8 -threads 2 -gemm-workers 2 \
  -resident-mib 1700 -cache-mib 8
```

Startup emits `ready`. Send one UTF-8 JSON object per line:

```json
{"id":"request-1","text":"The evidence is insufficient, Captain."}
```

Optional `frames` requests a single utterance with an explicit target duration
at 25 frames/second, bounded by the process `-frames` capacity:

```json
{"id":"fixed-duration","text":"A evidência é insuficiente, capitão.","frames":125}
```

Launch with `-frames 125` or greater for that request. Zero/omitted `frames` uses
automatic planning; explicit values bypass the shorter-first policy. Inputs must
still fit the 512-position prompt limit. Cache keys include target frames, so
different durations never reuse the same cached audio. This is useful when the
automatic estimator compresses pronunciation; it does not control regional accent.

Each request emits `start`, one `chunk` event per completed WAV, then `done`.
The `chunk.path` file is closed and usable before its event is written, and that
event is written before the next chunk starts inference. Consume events as they
arrive and play files in `index` order. Writes are synchronous; a slow reader
applies backpressure. Timing measures file availability, not actual playback.

A fresh private `omnivoice-*` directory under `output-dir` holds all files from
that process. Filenames use server sequence/index numbers, never request IDs.
IDs may repeat. WAVs are synthetic PCM16 mono at 24 kHz with 5 ms edge fades and
100 ms trailing silence except after the last chunk. Peak limiting is per chunk;
this can differ from globally limiting a joined long WAV. No final waveform is
accumulated in RAM. The caller owns file retention and cleanup; disk usage grows
with completed chunks even when the in-memory cache is disabled.

## Fixed process configuration

The model/reference, language, instructions, steps, denoise/postprocess settings,
frame caps and worker/residency policy are fixed at launch. Requests accept only
`id`, `text`, and optional `frames`. Restart to change voice or generation settings. Raw reference
encoding is performed beforehand through the existing `encode-reference` mode.

- `-weights-gguf`: optional matching GGUF backbone; codec assets stay in `-model`.
- `-resident-mib`, `-prepack`, `-gemm-workers`: existing CPU SIMD options.
- `-cache-mib 0..256`: process-local waveform payload budget; zero disables it.
- `-first-frames N`: optional first-chunk cap, `1..frames`; zero preserves normal
  planning. Later chunks use `-frames`. Text/whitespace reconstruction is exact.
- `-direct-q8` is not wired into serve; Q8 uses dequantisation plus SIMD GEMM.

The LRU cache holds at most 256 entries and copies successful, validated raw
chunk waveforms. Keys include exact chunk text and target frame count. Model,
reference and all other generation parameters are fixed for that cache lifetime.
Boundary fades/gaps are applied to a fresh copy for each output. Hits still write
new output files. The payload budget excludes bounded map/key bookkeeping.

Backbone capacity is 512 conditional positions and `frames` unconditional
positions; codec and generation buffers reserve `frames`. Prepared core compute
reuses buffers. Tokenisation, planning, result copies, JSON, output files and
cache insertion still allocate. This worker is not allocation-free end-to-end.

## Failure and cancellation

Malformed JSON, unknown fields, invalid requests and generation failures emit
`error`; subsequent valid lines can continue. `retained_chunks` reports files
created for that request, including files retained when a later chunk fails.
Previously emitted files remain usable. No invalid waveform enters the cache.

A line over 64 KiB or stdin read failure emits a fatal `error` and exits. A broken
stdout terminates the worker. SIGINT/SIGTERM cancels inference and closes stdin
to unblock an idle read; already emitted files remain. Individual SIMD calls and
filesystem writes finish before cancellation is observed. There is no per-request
cancel command and no concurrent inference. EOF exits cleanly after queued lines.

## Timings and measured behaviour

`ready` reports startup and reference/tokenizer, weight-open, codec-load and
runner-setup intervals. Runner setup includes resident/prepacked cache creation.
Chunk events report preparation, denoising, codec decode, postprocessing and
write durations, plus elapsed time since request start. `done` reports time to
first chunk and total request time. Cache hits have zero compute-stage durations.

N100, two vCPUs, four steps, two workers, resident float32 cache, same fixed voice:

| Experiment | Startup | First chunk after request | Request total | Audio |
|---|---:|---:|---:|---:|
| First cap 40, cold cache | 7.96 s | 20.09 s | 71.45 s | 5.32 s / 3 chunks |
| Same request, all cache hits | amortised | 1.34 ms | 2.95 ms | identical files |
| Normal cap 75, cache disabled, request 1 | 1.97 s | 19.38 s | 45.93 s | 4.18 s / 2 chunks |
| Same process, cache disabled, request 2 | amortised | 24.41 s | 50.83 s | identical files |

These separate runs have noisy setup/compute timings. They establish safe reuse
and cache behaviour, not a consistent warm-compute or short-first speedup. Short
chunks alter duration estimates and boundaries: the first-cap experiment split
after “The evidence is”. Listening acceptance is still required, and short-first
stays disabled by default. Cache retrieval is not new speech synthesis.

Evidence: `/workspace/tmp/omnivoice-serve-v1.jsonl`,
`omnivoice-serve-uncached-v2.jsonl`, `omnivoice-serve-tests.log` and private output
directories named in those events. Tests cover progressive emission, cache
ownership/LRU/limits, repeat-file equality, failure lifecycle, oversized input,
output failure, cancellation and first-limit Unicode/text preservation. Affected
tests/vet, race, no-CGo and Linux ARM64 builds pass. No system service was deployed.
