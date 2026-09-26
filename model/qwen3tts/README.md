# model/qwen3tts

Qwen3 TTS (text-to-speech): a talker/decoder pipeline that turns text + a speaker
reference into semantic tokens, acoustic frames, and a waveform. Like `model/lfm2`,
it is organized around per-stage **contracts** plus the satisfying runtime.

## Bounded CPU execution

The 0.6B CustomVoice CPU path supports a hard cap of64 acoustic frames through
`GenerateCappedGreedyCPU`, `GenerateCappedGreedyMinTwoCPU` and
`GenerateCappedSeededCPU`. `BoundedCPUResult.StoppedAtEOS` is true only when a
continuation actually selects EOS before the cap; EOS has no acoustic frame.
False on success means cap exhaustion, not a prediction about the next token.
First-token EOS remains an error with no partial output.

The pinned “Hi”/Ryan/English/seed42 case now reproduces native natural EOS after
46 frames, all736 codes exact versus Rust/Candle and waveform error below1.6e-6.
See the [qualification report](../../docs/validation/qwen3-tts-natural-eos-20260926.md).
This does not qualify arbitrary prompts, released64-frame exhaustion, streaming,
NVIDIA,1.7B or general voice quality. The fixed2/3/4-frame APIs remain unchanged.

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
