# Commands

Run commands from the repository root. Each command's `-h` output is the source for its current flags; model paths below refer to local assets, not automatic downloads.

| Task | Entry point | Guide |
|---|---|---|
| Download checkpoint assets | `scripts/download_models.py`, `make models-list` | [Model assets](model-assets.md) |
| Generate text, chat or serve an OpenAI-compatible API | `cmd/llm/llmgen`, `cmd/llm/llmchat`, `cmd/llm/llmserver` | [LLM commands](llm-commands.md) |
| Inspect GGUF and validate cache accounting | `cmd/models/ggufinspect`, `cmd/models/ggufsmoke` | [GGUF validation](gguf-validation.md) |
| Inspect incomplete model families without claiming inference support | `cmd/minicpmvinspect`, `cmd/qwen/qwen3ttsinspect`, `cmd/models/lfm2inspect` | [Model inspection](model-inspection.md) |
| Transcribe, translate or add speaker labels | `cmd/audio/whisper`, `cmd/audio/diarize-vtt`, `cmd/audio/moss-transcribe` | [Speech commands](speech-commands.md) |
| Score variable choices and compare native decision models | `cmd/jevlike`, `cmd/jevcompare` | [Jevlike](../../model/jevlike/README.md), [native-port bake-off](../experiments/jev-port-bakeoff-report-20260921.md) |
| Run/train Needle 3/2; Needle 3 archive text/head inference | `cmd/needle` | [Needle limits and validation](../../model/needle/README.md); [head training and width slicing](needle-head-training.md); [bounded tool calls](needle-tool-calls.md) |
| Extract entities, classes, relations or records | `cmd/gliner2` | [GLiNER 2.5](../../model/gliner2/README.md) |
| Run block-diffusion text generation | `cmd/diffusiongemmarun`, `cmd/diffusiongemmaserver` | [DiffusionGemma](../models/diffusiongemma/README.md) |
| Generate images | `cmd/image/ideogram4gen` | [Ideogram 4](../models/ideogram4-support.md) |

## Native scorers and extraction

GLiNER loads the local published checkpoint and emits JSON. Schema JSON enables classification, relation, record and mixed tasks; the model guide documents the supported fields and decode limits.

```bash
go run ./cmd/gliner2 -model /path/to/gliner2.5-base-v1 \
  -text 'Alice works at Acme in Lisbon.' -label person -label organization -label location
```

Jevlike's synthetic training path needs no external backbone. Frozen training/prediction needs local safetensors, configuration and tokenizer assets.

```bash
go run ./cmd/jevlike synthetic -output-dir /tmp/jevlike-data -train 64 -validation 16 -test 16
go run ./cmd/jevlike train -train /tmp/jevlike-data/train.jsonl \
  -validation /tmp/jevlike-data/validation.jsonl -checkpoint /tmp/jevlike.json -epochs 3
go run ./cmd/jevlike eval -checkpoint /tmp/jevlike.json -data /tmp/jevlike-data/test.jsonl
```

Reusable visual/action encoders are library APIs; the Jevlike CLI does not launch games, record sessions or orchestrate demonstrations.

## Jevlike experiment commands

These are distinct paths, not interchangeable benchmarks. Use the subcommand's
`-help` before providing local files; merely reading help does not load weights.

| Work | Commands | Reproduction/results |
|---|---|---|
| Prepare pinned grouped data | `prepare` | [Experiment preparation](../experiments/jevlike-qwen3/README.md) |
| Direct selected-label scoring/calibration | `score`, `direct-bench`, `direct-report`, `direct-calibrate` | [Direct study](../experiments/jevlike-qwen3/direct-study.md) |
| Feature materialisation/offline heads | `cache-extract`, `feature-score`, `cache-compare`, `cached-similarity`, `cached-train`, `cached-score`, `head-calibrate` | [Frozen-head study](../experiments/jevlike-qwen3/frozen-head-study.md) |
| Fresh/reused candidates and permutations | `head-online-bench`, `head-permutation`, `score-batch` | [Head results](../experiments/jevlike-qwen3/frozen-head-study.md), [candidate contract](../experiments/jevlike-qwen3/candidate-contract.md) |
| Actual transformer prefix reuse | `prefix-bench` (plus library prefix APIs) | [Prefix study](../experiments/jevlike-qwen3/prefix-study.md) |
| Immutable held-out evaluation/reporting | `final-eval`, `final-report` | [Frozen policy](../experiments/jevlike-qwen3/final-evaluation-policy.md), [current status](../experiments/jevlike-qwen3/status-report-20260919.md) |

`final-eval` requires the frozen manifests and hashes, verifies every existing
record, and evaluates only missing rows. `-plan` performs admission, identity,
cache-bound and disk-space checks without loading the GPU encoder. `final-report`
requires the complete unlocked record set and applies the existing calibrations
without refitting. The frozen run completed 1,440/1,440 rows; its [final
report](../experiments/jevlike-qwen3/final-report-20260921.md) includes exact
commands and results. Do not replay or replace completed records, change the
frozen recipe, or use the final-test outputs for tuning.

`cmd/jevcompare` runs the separate [native JEV-equivalent bake-off](../experiments/jev-port-bakeoff-report-20260921.md). It maps a frozen choice request to OpenJEV, Decider, Laya or Nimble, writes strict resumable JSONL, and can capture a scoring CPU profile. Supplying `-report-results` and `-report-output` skips model loading and verifies the frozen cohort/results before writing accuracy, random baseline, NLL, Brier, ECE, reliability and risk/coverage metrics. The report normalizes stored candidate scores within each offered set and never fits a temperature.

## Checks and hardware

Use [validation gates](../validation/validation-gates.md) for host checks, model fixtures and compile-only cross-builds. K3/SpacemiT diagnostic commands require the board's Linux/RISC-V environment; an amd64 or ARM64 build is not permission to run its IME or TCM operations.

[Runtime tuning](tuning.md) covers placement, KV caches and generation controls. [Supported models](../models/supported-models.md) distinguishes inference paths from metadata-only tools.
