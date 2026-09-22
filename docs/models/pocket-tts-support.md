# Pocket TTS

`model/pockettts` targets Kyutai Pocket TTS with native Go and repository SIMD kernels. The supported path does not call Torch, ONNX Runtime, CGo model runtimes or Python subprocesses.

## Pinned sources

- Upstream source: `kyutai-labs/pocket-tts@0acce6b2f390150267557770d2098c5caa9a18ac`.
- Public English no-voice-cloning weights: `kyutai/pocket-tts-without-voice-cloning@e7205b6ee50e654a5ea19f0e9df2b0813b05e921`, SHA-256 `916ccd2686e9311cb40054893a3c4284393d658825ffc714a276f3e9b152344f`, 219,029,196 bytes.
- English tokenizer: revision `00eac05ed3d16bdc3f6b5d598874019c34a89214`, SHA-256 `f498428e1eafee50492f7be13dc9bfafcfc12e508cd0eb1b01c92ecd5d8c6687`, 245,020 bytes.
- Upstream code uses the MIT licence. The model card reports CC-BY-4.0 and a prohibited-use policy. The full voice-cloning bundle requires accepting the Hugging Face access terms.

## Architecture

The released English student contains 214 BF16 tensors:

- a 4,000-piece Unigram tokenizer with always-prepended metaspace and UTF-8 byte fallback;
- six FlowLM transformer layers, width 1,024, 16 heads and 4,096-wide feed-forward blocks;
- a 32-dimensional autoregressive latent stream at 12.5 Hz;
- a six-block, 512-wide `SimpleMLPAdaLN` flow head with two LSD time conditions;
- a streaming Mimi encoder/decoder with two 512-wide transformer layers, SEANet ratios `[6,5,4]`, a 16× latent-rate resampler and 24 kHz mono output;
- exactly 1,920 PCM samples per generated latent frame.

Training uses a frozen Mimi codec, aligned transcript/audio manifests, optional latent precomputation, FlowLM flow and EOS losses, AdamW and EMA. The reference recipe trains a 24-layer teacher and then distils its conditioned/null backbone states into the six-layer student while keeping the flow and EOS heads frozen.

## Implemented

- Strict YAML topology parsing and 12.5 Hz/24 kHz geometry validation.
- Strict tensor inventory and released-checkpoint shape/dtype validation.
- Native Unigram/Metaspace/byte-fallback tokenisation with pinned upstream fixtures for whitespace, newline, accents and unsupported Unicode.
- Owned-F32 six-layer FlowLM stateless and request-owned streaming-KV prefill using SIMD LayerNorm, GEMV, causal attention, interleaved-pair RoPE and tanh-GELU. The pinned released fixture matches PyTorch 2.13 within `3e-5` for the final hidden row and EOS logit; streaming and stateless paths agree within the same bound.
- Native import of the public, revision-pinned Alba voice-state K/V safetensors. Text prompting, BOS hidden state, EOS logit, fixed-noise flow velocity and the first latent match the independent upstream fixture.
- SIMD affine and LSD integration primitives with scalar differential tests.
- Owned-F32 SIMD `SimpleMLPAdaLN` forward execution, including the upstream unbiased-variance time-embedding normalisation quirk.
- Independent released-weight flow-head parity against a PyTorch 2.13 fixture with maximum accepted absolute error `2e-4`.
- Stateful native Mimi latent projection, depthwise 16× resampling, two-layer finite-context transformer and SEANet decoder. A released first-latent fixture emits exactly 1,920 samples and matches the upstream waveform within `3e-5`.
- Autoregressive generation orchestration with caller-owned noise, EOS countdown, persistent FlowLM/Mimi state, exact 1,920-sample chunks and exclusive PCM16 mono WAV output. A released one-frame fixture validates voice-state → text → FlowLM → LSD → Mimi → PCM as one contract.
- Session-owned `StepInto`, `DecodeFrameInto` and `GenerateInto` paths perform zero warm heap allocations. The measured Intel i7-12700 path emits the first 80 ms chunk in about 51–54 ms, 400 ms of audio in 143–147 ms, and 2 seconds in 398–427 ms. See [the validation record](../validation/pocket-tts-native-inference-2026-09-22.md).
- `cmd/audio/pockettts` loads external config, model, tokenizer and voice-state files and writes a seeded WAV without Torch, ONNX Runtime, CGo model runtimes or Python subprocesses.

## Next inference slices

1. Add raw-audio Mimi encoding for voice cloning after the gated bundle and an approved prompt fixture are available.
2. Run native ARM64 and RVV performance qualification; amd64 allocation and performance targets pass on the recorded i7-12700 host.

## Native training

The first deterministic training gate is implemented:

- aligned JSONL admission validates finite monotonic word timings and requires a prompt/target cut with at least one second on each side;
- EOS reduction covers valid frames plus exactly the first invalid frame;
- FlowMatching and normalized LSD diagonal losses expose analytic output gradients;
- a frozen-backbone affine Flow/EOS topology has independent PyTorch parity for loss, every parameter gradient, one AdamW update and EMA within `3e-7`;
- F32-owned `SimpleMLPAdaLN` reverse-mode covers every parameter, latent input, condition and both time inputs; exact forward-mode time JVPs and reverse-over-JVP mixed derivatives cover both time conditions; the normalized LSD `s→t` row implements the upstream minimal stop-gradient endpoint rule; pinned upstream-module fixtures match within `3e-5`, with separate all-parameter central differences;
- F32-owned stateless transformer backward covers bounded causal attention, adjacent-pair RoPE, softmax, residual layer scales, both affine LayerNorms, tanh-GELU FFN and final LayerNorm; a context-two upstream fixture matches output, full sequence gradient and every parameter within `1e-5`;
- the exact one-row conditioning layout covers BOS-before-voice, voice projection, text lookup, shifted `[BOS,audio[:-1]]` projection, gathered audio rows and EOS; the upstream layout fixture matches the assembled sequence, outputs, caller-input gradients and every parameter within `2e-5`;
- versioned checkpoint state stores parameters, AdamW settings/moments, EMA and step through a directory-durable atomic save, and resumed second-step state is byte-for-byte equivalent to uninterrupted state.

See [the training validation record](../validation/pocket-tts-native-training-2026-09-22.md) for the fixture and commands.

The next slices are:

1. combine EOS, normalized LSD diagonal and `s→t` gradients for complete one-step parity;
2. frozen Mimi latent precomputation;
3. released-format checkpoint export;
4. 24-layer teacher to six-layer depth/CFG distillation;
5. long CPU qualification. GPU training is a separate backend task.

Preset-voice inference is complete for the pinned released artefacts. Raw-audio voice cloning needs the gated encoder bundle. Full native training is incomplete.
