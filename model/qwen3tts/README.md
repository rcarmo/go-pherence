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
A separate sentence exhausts all 64 frames with EOS enabled: native CPU matches
all 1,024 codes and its 122,880-sample waveform within `1.6e-6` against the
independent Rust/Candle reference. The [64-frame qualification](../../docs/validation/qwen3-tts-cap64-parity-20260926.md)
records the numerical gate and same-instance recovery. This does not establish
that the sentence finishes at the cap. Arbitrary prompts, streaming, NVIDIA,
1.7B and general voice quality remain unqualified. A separate [four-caller
short CPU check](../../docs/validation/qwen3-tts-four-caller-cpu-20260926.md)
passed with one shared checkpoint; more callers and long-running concurrency
are unqualified. `GenerateCappedSeededCPUContext` adds cooperative cancellation
between stages with no partial output; [the bounded cancellation check](../../docs/validation/qwen3-tts-capped-cancellation-20260926.md)
uses synthetic and released two-frame recovery. Kernels are non-interruptible,
so cancellation latency and long-request cancellation are unqualified. The
fixed 2/3/4-frame APIs remain unchanged.

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
