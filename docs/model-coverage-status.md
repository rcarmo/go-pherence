# Model coverage status

This page is the short-form status for the current non-LLM model-family coverage work. The detailed roadmaps remain in [qwen3-tts-support.md](qwen3-tts-support.md) and [lfm2-moe-support.md](lfm2-moe-support.md).

## Machine-readable manifest

`model-coverage-manifest.json` tracks the same coverage gates in a compact form for CI and status tooling. Keep it in sync when a model-family coverage layer moves from `false` to `true`.

Summarize it with:

```bash
make model-coverage
make model-coverage-json
make model-coverage-pending MODEL_COVERAGE_FAMILY=qwen3_tts
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -json
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -family qwen3_tts -pending-only
```

## Validation command

Use the focused model-coverage target while these families are in metadata/fixture/inspector stages:

```bash
make test-model-coverage
```

It validates:

- `docs` manifest tests
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
- `model/qwen3tts/prefill.go` — Talker prefill stream/embedding sizing contract.
- `model/qwen3tts/talker_input.go` — text projection plus codec-control embedding fusion layout.
- `model/qwen3tts/prompt_runtime.go` — prompt-specific prefill plus Talker input-fusion contract.
- `model/qwen3tts/embedding_layout.go` — text embedding, projection, codec-head, and codec-embedding matrix sizing.
- `model/qwen3tts/capabilities.go` — Base, CustomVoice, and VoiceDesign conditioning contracts.
- `model/qwen3tts/speaker_language.go` — CustomVoice speaker/native-language compatibility metadata.
- `model/qwen3tts/speaker_encoder.go` — optional speaker encoder embedding/sample-rate/reference-audio layout.
- `model/qwen3tts/tensors.go` — tensor group readiness.
- `model/qwen3tts/tensor_shapes.go` — safetensors dtype/shape summaries.
- `model/qwen3tts/tensor_shape_validation.go` — config-aware shape sanity checks.
- `model/qwen3tts/pipeline.go` — conditioning → prompt prefill → Talker → CodePredictor → Decoder12Hz stage plan.
- `model/qwen3tts/attention_layout.go` — Talker and CodePredictor GQA/RoPE/RMSNorm attention contracts.
- `model/qwen3tts/ffn_layout.go` — Talker and CodePredictor gated-MLP projection sizing contracts.
- `model/qwen3tts/semantic.go` — Talker semantic token stream layout and range validation.
- `model/qwen3tts/frame.go` — semantic-group plus 15-code acoustic frame layout validation.
- `model/qwen3tts/code_predictor_heads.go` — 15 acoustic-head logits layout and validation contract.
- `model/qwen3tts/decoder_input.go` — acoustic-codebook tensor contract passed into Decoder12Hz.
- `model/qwen3tts/waveform.go` — Decoder12Hz mono 24kHz waveform/WAV sizing contract.
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
- `model/lfm2/routing.go` — expert count, active top-k, normalization, bias, and routed-scaling contract.
- `model/lfm2/router_layout.go` — router projection/logits/top-k scratch sizing contract.
- `model/lfm2/ffn_layout.go` — dense and routed expert FFN dimension/parameter contract.
- `model/lfm2/norm.go` — RMSNorm epsilon, vector count, and scratch sizing contract.
- `model/lfm2/embedding_layout.go` — token embedding, tied/untied LM-head, and byte sizing contract.
- `model/lfm2/conv_state.go` — per-conv-layer cache layout and byte sizing contract.
- `model/lfm2/conv_projection.go` — per-conv-layer short-convolution kernel/bias sizing contract.
- `model/lfm2/attention_kv.go` — full-attention layer KV-cache layout and byte sizing contract.
- `model/lfm2/attention_projection.go` — full-attention Q/K/V/O projection and GQA sizing contract.
- `model/lfm2/context.go` — vocabulary, max-context, tied-embedding, and RoPE context contract.
- `model/lfm2/rope.go` — RoPE theta/head-dim/full-attention position contract.
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
