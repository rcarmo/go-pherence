# Model coverage status

This page is the short-form status for the current non-LLM model-family coverage work. The detailed roadmaps remain in [qwen3-tts-support.md](qwen3-tts-support.md) and [lfm2-moe-support.md](lfm2-moe-support.md).

## Validation command

Use the focused model-coverage target while these families are in metadata/fixture/inspector stages:

```bash
make test-model-coverage
```

It validates:

- `loader/safetensors`
- `model/qwen3tts`
- `model/lfm2`
- `cmd/qwen3ttsinspect`
- `cmd/lfm2inspect`

Whole-tree dry compiles are still useful, but currently fail in unrelated experimental Spacemit IME2 and `tmp/diarize` packages.

## Qwen3-TTS

Status: metadata, tokenizer/prompt, fixture scaffold, tensor readiness, shape validation, runtime sizing, and inspector coverage are implemented. Audio generation is not implemented.

Implemented package/command surface:

- `model/qwen3tts/config.go` — variant detection and parsed talker/code-predictor dimensions.
- `model/qwen3tts/tokens.go` — ChatML, TTS, codec, language, and speaker constants.
- `model/qwen3tts/prompt.go` — tokenizer loading plus tokenized CustomVoice prompt builder.
- `model/qwen3tts/capabilities.go` — Base, CustomVoice, and VoiceDesign conditioning contracts.
- `model/qwen3tts/tensors.go` — tensor group readiness.
- `model/qwen3tts/tensor_shapes.go` — safetensors dtype/shape summaries.
- `model/qwen3tts/tensor_shape_validation.go` — config-aware shape sanity checks.
- `model/qwen3tts/shapes.go` — Talker, CodePredictor, and 12Hz decoder runtime sizing plan.
- `model/qwen3tts/fixtures.go` and `testdata/` — small committed prompt fixture schema.
- `cmd/qwen3ttsinspect` — config/tensor/shape/capability/runtime-plan/tokenized-prompt inspector.

Useful commands:

```bash
make models-list MODEL_DOWNLOAD_FLAGS='--group qwen3tts'
make qwen3tts-inspect \
  QWEN3TTS_MODEL=models/qwen3-tts-0.6b-customvoice \
  QWEN3TTS_TEXT="Hello world"
```

Strict checkpoint validation:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen3ttsinspect \
  -model models/qwen3-tts-0.6b-customvoice \
  -text "Hello world" \
  -strict -json
```

Remaining coverage before runtime implementation:

- Capture qwen3-tts-rs reference values for first semantic token, one acoustic frame, and a short decoded WAV summary.
- Add fixture summaries for Base and VoiceDesign conditioning once their prompt/reference inputs are confirmed.

Runtime work still pending:

- CPU Talker path.
- CPU CodePredictor path with short KV cache.
- Decoder12Hz and WAV output.
- NVIDIA acceleration and streaming after CPU/reference parity.

## LFM2.5-8B-A1B

Status: metadata, tensor readiness, shape validation, layer schedule, runtime state sizing, fixture scaffold, download registration, and inspector coverage are implemented. Generation is not implemented.

Implemented package/command surface:

- `model/lfm2/config.go` — strict `lfm2_moe` config parsing and validation.
- `model/lfm2/tensors.go` — tensor group readiness.
- `model/lfm2/tensor_shapes.go` — safetensors dtype/shape summaries.
- `model/lfm2/tensor_shape_validation.go` — config-aware shape sanity checks.
- `model/lfm2/schedule.go` — explicit conv/full-attention layer schedule.
- `model/lfm2/execution.go` — per-layer dense-vs-routed-MoE execution role plan.
- `model/lfm2/state.go` — conv state, full-attention KV, and MoE sizing plan.
- `model/lfm2/fixtures.go` and `testdata/` — committed metadata fixture.
- `cmd/lfm2inspect` — config/tensor/shape/schedule/runtime-plan inspector.

Useful commands:

```bash
make models-list MODEL_DOWNLOAD_FLAGS='--group lfm2'
make lfm2-inspect LFM2_MODEL=models/lfm2.5-8b-a1b
```

Strict checkpoint validation:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/lfm2inspect \
  -model models/lfm2.5-8b-a1b \
  -strict -json
```

Remaining coverage before runtime implementation:

- Capture Transformers reference summaries for tokenization, first-token logits, one conv layer output, one full-attention layer output, router top-k, and one expert output.
- Extend tensor shape validation after real checkpoint tensor names are audited for all MLP/expert projections.

Runtime work still pending:

- CPU embedding/RMSNorm/full-attention path.
- LFM convolution/state block with `conv_L_cache=3` semantics.
- MoE router/top-k/expert FFN parity.
- NVIDIA/local optimization only after CPU/reference parity.
