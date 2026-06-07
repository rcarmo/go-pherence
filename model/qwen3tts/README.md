# model/qwen3tts

Qwen3 TTS (text-to-speech): a talker/decoder pipeline that turns text + a speaker
reference into semantic tokens, acoustic frames, and a waveform. Like `model/lfm2`,
it is organized around per-stage **contracts** plus the satisfying runtime.

| Area | Files |
|---|---|
| Pipeline / runtime | `pipeline.go`, `pipeline_contract.go`, `prefill.go`, `capabilities.go`, `runtime_*.go` |
| Prompt / speaker | `prompt.go`, `prompt_runtime.go`, `speaker_encoder.go`, `speaker_language.go`, `tokens.go` |
| Talker / decoder | `talker_contract.go`, `talker_input.go`, `decoder_contract.go`, `decoder_input.go` |
| Code predictor | `code_predictor_contract.go`, `code_predictor_heads.go` |
| Semantic → audio | `semantic.go`, `frame.go`, `waveform.go` |
| Layout | `attention_layout.go`, `embedding_layout.go`, `ffn_layout.go`, `shapes.go` |
| Inspection / readiness | `readiness.go`, `tensors.go`, `tensor_shapes.go`, `tensor_shape_validation.go`, `fixtures.go` |

> The readiness/tensor-shape inspection layer is structurally shared with
> `model/lfm2` (specialized via package-local types).
