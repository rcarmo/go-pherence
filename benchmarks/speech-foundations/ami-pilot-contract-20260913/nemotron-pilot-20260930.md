# Nemotron AMI ES2004a headset-mix pilot — 30 September 2026

At revision `ccb0f3429b2620af184ad6b265b6d7c3175ff313`, all three local backends produced the same transcript and speaker spans for the pinned 60-second AMI excerpt. The labelled pilot has 38.55% transcript-only WER and 54.25% diarization error rate (DER) with a zero collar. One meeting excerpt gives no corpus-level quality estimate.

## Inputs and method

[The AMI source contract](README.md) pins the official CC BY 4.0 headset-mix WAV and annotation ZIP and describes local preparation. The local `manifest.json` matched [`pilot-result.json`](pilot-result.json); the excerpt WAV, RTTM and word JSON had their pinned SHA-256 values. None of the AMI media, annotations, RTTM or reference words is committed. The evaluation interval is 350–410 seconds of `ES2004a`; the excerpt contains 960,000 samples at 16 kHz, 28 RTTM turns (four speakers) and 179 timed lexical words.

Checkpoint SHA-256: ASR `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174`, tokenizer `3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246`, diarization `c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6`. These hashes pin model assets, not numerical acceptance. Runs used Go 1.26.3 on an Intel i7-12700 with RTX 3060 (driver 610.57.04), `GOMAXPROCS=4`. Backend outputs happened to be identical in this run; output hashes below identify the saved files for replay and do not define an exact-match gate for future implementations.

From the repository root, with locally prepared data and checkpoints:

```sh
p=/workspace/tmp/nemotron-ami-es2004a-350-410
for backend in simd ptx vulkan; do
  GOMAXPROCS=4 go run ./cmd/audio/nemotron -task asr -backend "$backend" \
    -input "$p/ES2004a-350-410.wav" -model checkpoints/nemotron/asr/model.safetensors \
    > "$p/nemotron-$backend-asr.txt" 2> "$p/nemotron-$backend-asr.log"
  tower=
  case "$backend" in ptx) tower=-ptx-tower;; vulkan) tower=-vulkan-tower;; esac
  GOMAXPROCS=4 go run ./cmd/audio/nemotron -task diarization -backend "$backend" $tower \
    -input "$p/ES2004a-350-410.wav" -model checkpoints/nemotron/diarization/model.safetensors \
    > "$p/nemotron-$backend-diarization.json" 2> "$p/nemotron-$backend-diarization.log"
done
```

| Backend | ASR request | Diarization request | Scope |
|---|---:|---:|---|
| SIMD | 34.303 s | 38.238 s | CPU |
| PTX | 34.816 s | 12.627 s | ASR projection hybrid; diarization opt-in resident tower hybrid |
| Vulkan | 35.646 s | 16.328 s | ASR projection hybrid; diarization opt-in resident tower hybrid |

CLI `request_elapsed` excludes WAV/checkpoint load and Go build, but includes inference GPU setup, transfers and teardown. Each request is below the 60-second audio duration on this host. These are single runs on a potentially busy machine; they do not establish sustained or portable throughput. The resident GPU diarization paths retain the frontend, audio layer 0, head and speaker cache on CPU; GPU ASR offloads only subsampling projection.

A separate binary (built once with `go build -o "$p/nemotron-cli" ./cmd/audio/nemotron`) was timed for each task/backend with `GOMAXPROCS=4 /usr/bin/time -f 'wall_elapsed=%e maxrss_kb=%M' -o "$p/wall-$backend-$task.time" "$p/nemotron-cli"` followed by the same task/backend/tower/input/model flags above, with stdout and stderr redirected to local files. Wall time includes WAV and checkpoint load, inference and process exit; it excludes compilation. Each re-run preserved the prior transcript text and 39 speaker-span geometries.

| Backend | ASR process wall / peak RSS | Diarization process wall / peak RSS |
|---|---:|---:|
| SIMD | 37.38 s / 5,155,940 KiB | 38.77 s / 1,581,632 KiB |
| PTX | 38.40 s / 5,273,564 KiB | 13.07 s / 1,687,052 KiB |
| Vulkan | 38.35 s / 5,371,128 KiB | 16.96 s / 2,720,788 KiB |

Four further process invocations per task/backend used the same binary, input and flags, in interleaved backend/task order (`simd`, `ptx`, `vulkan` within ASR and then diarization). Their stdout was discarded; each exited successfully. Across five samples per case, process-wall medians and ranges were:

| Backend | ASR median [min, max] | Diarization median [min, max] |
|---|---:|---:|
| SIMD | 37.38 [36.80, 38.07] s | 38.31 [38.04, 39.28] s |
| PTX | 37.93 [37.39, 38.40] s | 13.07 [12.96, 13.23] s |
| Vulkan | 38.13 [37.94, 39.02] s | 16.78 [16.64, 16.96] s |

All 30 load-inclusive invocations finished within 60 seconds on this host, and the saved-output replay matched across the first six. The GPU projection ASR hybrids show no supported wall-time speedup over SIMD because their ranges overlap. These are separate short cold-process runs; no long-lived stream, sustained multi-request workload, long soak or other host has been measured.

## Saved output and scoring

All three saved transcripts contain 119 normalized words. Their transcript file SHA-256 is `61e0b6db19a3ae4ec79ff6733690b5b52b941c25361bd61c1536b0779e45a3ef`. All three span files contain 39 turns; their SHA-256 is `e833993911c7c6e5294764ed896b1c32d60ab1fc07092f87763d69351f8aa8cd`.

`score_nemotron_ami_pilot.py` checks the local manifest against the committed contract, checks each reference asset hash and its geometry, and validates saved Go speaker spans. It reuses the repository's NFKC/casefold Unicode alphanumeric-plus-internal-apostrophe tokenisation and word edit counts. Hypothesis text has no word timestamps or speaker assignments; cpSA-WER is unavailable. DER/JER use `pyannote.metrics` with `[0,60]` seconds as the UEM, `skip_overlap=False`, and collars of 0 and 0.25 seconds. The two collars are reported separately; no score is a pass/fail threshold.

```sh
p=/workspace/tmp/nemotron-ami-es2004a-350-410
# External scoring environment; no project dependency change:
uv venv /workspace/tmp/nemotron-ami-metrics
uv pip install --python /workspace/tmp/nemotron-ami-metrics/bin/python \
  'pyannote.metrics==4.1' 'pyannote.database==6.1.1'
for backend in simd ptx vulkan; do
  PYTHONDONTWRITEBYTECODE=1 /workspace/tmp/nemotron-ami-metrics/bin/python \
    scripts/score_nemotron_ami_pilot.py \
    --contract benchmarks/speech-foundations/ami-pilot-contract-20260913/pilot-result.json \
    --manifest "$p/manifest.json" --audio "$p/ES2004a-350-410.wav" \
    --rttm "$p/ES2004a-350-410.rttm" --words "$p/ES2004a-350-410.words.json" \
    --transcript "$p/nemotron-$backend-asr.txt" \
    --spans "$p/nemotron-$backend-diarization.json" \
    --output "$p/nemotron-$backend-reviewed-score.json"
done
```

The [committed PTX score JSON](nemotron-pilot-score-20260930.json) contains the checked reference and saved-output hashes, metric details and `qualified: false`. The same scores were obtained independently for each saved backend output:

| Measure | Zero collar | 0.25-second collar |
|---|---:|---:|
| WER | 69 / 179 = 0.385475 (7 substitutions, 61 deletions, 1 insertion) | — |
| DER, overlap included | 0.542506 | 0.528448 |
| JER, overlap included | 0.596843 | 0.594789 |

At the zero collar, DER counts 40.208 seconds of missed speaker time, 4.233 seconds of confusion and no false alarm against 81.918 seconds of reference speaker time (overlap contributes multiple speaker-seconds). The 0.25-second collar changes the scored reference time to 60.216 speaker-seconds. A previous provisional 0.594789 figure belongs to **JER at the 0.25-second collar**, not DER; the metric names and collars are explicit here.

The scorer tests use small synthetic reference assets and a stubbed metric call to test hash/geometry checks, token counts, CLI-only scope and parameter forwarding. The real pilot scores above exercise the installed `pyannote` implementation. Further work needs more rights-reviewed labelled recordings, a pinned independent reference-system comparison, repeated transfer-inclusive and complete-request timing, cancellation/soak, and other hardware before quality or performance qualification.
