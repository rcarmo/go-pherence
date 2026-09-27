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
1.7B and general voice quality remain unqualified. Separate [four-caller](../../docs/validation/qwen3-tts-four-caller-cpu-20260926.md)
and [eight-caller](../../docs/validation/qwen3-tts-eight-caller-cpu-20260926.md)
short CPU checks passed on pinned mixed-input fixtures with one shared
checkpoint. Nine or more callers and long-running concurrency are unqualified.
`GenerateCappedSeededCPUContext` adds cooperative cancellation between stages
with no partial output; [the bounded cancellation check](../../docs/validation/qwen3-tts-capped-cancellation-20260926.md)
uses synthetic and released two-frame recovery. A [released peer check](../../docs/validation/qwen3-tts-cancel-peer-20260926.md)
cancels after Prefill while another pinned request completes on the same model.
A separate [sentence check](../../docs/validation/qwen3-tts-long-cancellation-20260926.md)
cancels after 32 complete frames and recovers a pinned 64-frame waveform on
the same loaded models. Kernels remain non-interruptible; hard cancellation
latency and concurrent long-request cancellation are unqualified. A separate
[two-hour retention run](../../docs/validation/qwen3-tts-two-hour-retention-20260927.md)
kept 26 pinned 32-frame outputs across 13 rounds and released them without
post-GC live-heap growth; days-long retention and broader inputs remain open.
Opt-in NVIDIA [input-projection](../../docs/validation/qwen3-tts-gpu-input-projection-20260927.md),
[output-projection](../../docs/validation/qwen3-tts-gpu-output-projection-20260927.md),
[causal pre-convolution](../../docs/validation/qwen3-tts-gpu-preconv-20260927.md)
and [decoder-initial convolution](../../docs/validation/qwen3-tts-gpu-decoder-initconv-20260927.md)
diagnostics compare four decoder stages against pinned Rust traces. They do
not change the CPU synthesis path or qualify GPU waveform generation. The
[pre-convolution benchmark](../../docs/validation/qwen3-tts-gpu-preconv-benchmark-20260927.md)
measures the fully host-staged diagnostic at 4.40× CPU stage time on one
released 64-frame input. A separate [resident-buffer trial](../../docs/validation/qwen3-tts-gpu-preconv-resident-20260927.md)
measures an owned-output pre-convolution stage at about 2.59× CPU speed with
prepared GPU weights and scratch. A [decoder-initial convolution resident-buffer trial](../../docs/validation/qwen3-tts-gpu-decoder-initconv-resident-20260927.md)
measures its 7-tap stage at about 3.02× CPU speed for the fixed input. Neither
measures full synthesis. An opt-in [two-convolution hybrid decoder check](../../docs/validation/qwen3-tts-gpu-decoder-hybrid-20260927.md)
passes the unchanged 64-frame waveform gate while recording drift at the live
second-stage Rust boundary. A separate [final-convolution GPU diagnostic](../../docs/validation/qwen3-tts-gpu-finalconv-20260927.md)
passes a fixed stage-error bound against the saved Rust input, with differing
GPU/CPU bits. A separate [live final-convolution waveform check](../../docs/validation/qwen3-tts-gpu-finalconv-live-20260927.md)
passes the unchanged 64-frame waveform gate on the CPU decoder's live input;
its upstream final-Snake tensor differs from the saved Rust trace. None of
these checks qualifies production GPU dispatch. A [three-convolution live decoder check](../../docs/validation/qwen3-tts-gpu-threeconv-20260927.md)
passes the same 64-frame waveform gate with pre-convolution, decoder-initial
and final convolutions on GPU; remaining stages and full-request admission
are unqualified. A [decoder-only timing comparison](../../docs/validation/qwen3-tts-gpu-threeconv-decoder-benchmark-20260927.md)
found nearly equal five-sample medians on one fixed input, with many more
Go allocations in the diagnostic GPU path. A [CPU decoder profile and GPU hold decision](../../docs/validation/qwen3-tts-gpu-admission-hold-20260927.md)
identifies Candle-order projection reductions as the largest sampled CPU cost;
GPU admission still needs live chained parity, full-request performance and
resource gates. An opt-in [batched ConvNeXt FC1 stage check](../../docs/validation/qwen3-tts-gpu-convnext-fc1-20260927.md)
matches the independent Rust trace bitwise with two Candle-order 512-wide
GPU reductions; a full-width SGEMM misses the fixed stage gate. It has no
live decoder waveform or production dispatch. The fixed 2/3/4-frame APIs
remain unchanged.

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
