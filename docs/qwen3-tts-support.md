# Qwen3-TTS support roadmap

This note maps <https://github.com/TrevorS/qwen3-tts-rs> and the official Qwen3-TTS checkpoints onto the current `go-pherence` repository, then lays out a staged native implementation plan.

## Source reviewed

- Reference implementation: <https://github.com/TrevorS/qwen3-tts-rs>
- Upstream model family: <https://github.com/QwenLM/Qwen3-TTS>
- Official checkpoint IDs referenced by the Rust implementation:
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
  - `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

The local reference clone used for this assessment is disposable review material under `/workspace/tmp/qwen3-tts-rs`; do not vendor it.

## Executive summary

Qwen3-TTS is a good fit for `go-pherence`, but it is a new multi-stage audio generation pipeline rather than a simple variant of the current Qwen text decoder. The project already has several reusable pieces:

- Qwen-family transformer code, RoPE/MRoPE-adjacent helpers, KV-cache patterns, sampling, and NVIDIA/CPU execution paths;
- `loader/safetensors`, `loader/tokenizer`, and config parsing patterns;
- `models/whisper` audio frontend/decoder infrastructure and GPU LM-head work;
- `models/speaker` ECAPA/SpeechBrain speaker embedding, fbank/mel extraction, smoothing, and validation commands;
- SIMD/NVIDIA kernels for RMSNorm, matmul/GEMV, activation, attention, and GPU-side logits operations.

The missing pieces are TTS-specific model ownership and an audio codec decoder/encoder. The recommended first target is **0.6B CustomVoice text-to-speech** because it avoids reference-audio conditioning and speaker-encoder parity while still exercises the full talker → code predictor → decoder path.

## Reference pipeline shape

The Rust implementation exposes a three-stage pipeline:

1. **TalkerModel**
   - Autoregressively generates semantic codec tokens from text.
   - Uses ChatML-style text input, dual text/codec embeddings, speaker/control prefix tokens, RMSNorm, GQA, and MRoPE.
   - Important token IDs from the reference:
     - ChatML: `IM_START=151644`, `IM_END=151645`, `ASSISTANT=77091`, `NEWLINE=198`.
     - TTS text tokens: `TTS_PAD=151671`, `TTS_BOS=151672`, `TTS_EOS=151673`.
     - Codec control: `CODEC_PAD=2148`, `CODEC_BOS=2149`, `CODEC_EOS=2150`, `CODEC_THINK=2154`, `CODEC_NOTHINK=2155`, `CODEC_THINK_BOS=2156`, `CODEC_THINK_EOS=2157`.
     - Codec vocabulary size: `3072`.
   - CustomVoice speaker token IDs:
     - Serena `3066`, Vivian `3065`, UncleFu `3010`, Ryan `3061`, Aiden `2861`, OnoAnna `2873`, Sohee `2864`, Eric `2875`, Dylan `2878`.
   - Language token IDs include English `2050`, Chinese `2055`, Japanese `2058`, Korean `2064`, German `2053`, French `2061`, Russian `2069`, Portuguese `2071`, Spanish `2054`, Italian `2070`.

2. **CodePredictor**
   - For each semantic token, generates 15 acoustic codec tokens, producing a 16-codebook frame `[semantic, acoustic_0..14]`.
   - Reference architecture: 5 decoder layers, 15 codec embeddings, 15 LM heads, standard RoPE, KV cache length around 17 tokens per generated frame.
   - 1.7B talker variants still use a 1024-hidden code predictor and add a projection when talker hidden size differs from code predictor hidden size.
   - Hot-loop optimizations in the Rust implementation worth copying conceptually: preallocated KV cache, cached token suppression mask, deferred device-to-host acoustic-code transfer, GPU-side sampling/repetition penalty, and fused residual+RMSNorm.

3. **Decoder12Hz / speech tokenizer**
   - Converts 16-codebook frames to 24 kHz mono waveform.
   - The Rust reference describes ConvNeXt blocks and transposed-convolution upsampling.
   - Base/ICL voice cloning additionally needs the paired codec encoder to convert reference audio into codec codes.

## Model variants and first target

| Variant | Size class | Conditioning | Recommended priority |
|---|---:|---|---:|
| 0.6B CustomVoice | ~1.8 GB | fixed speaker tokens | 1 |
| 0.6B Base | ~1.8 GB | reference audio x-vector/ICL | 3 |
| 1.7B CustomVoice | ~3.9 GB | fixed speaker tokens | 2 |
| 1.7B VoiceDesign | ~3.8 GB | text-described voice | 4 |
| 1.7B Base | ~3.9 GB | reference audio x-vector/ICL | 5 |

Start with `0.6B CustomVoice` because it avoids ECAPA/reference-codec front-end complexity. Then scale the same path to `1.7B CustomVoice`, add `VoiceDesign` prompt conditioning, and only then add Base voice cloning.

## Fit against current repository

### Reusable now

- `model/qwen` already owns Qwen3/Qwen3Next-style decoder logic, QKV layout tests, recurrent/full-attention parity work, MTP code predictor ideas, prompt-state caching, and NVIDIA MLX cache/planner experiments.
- `models/whisper` proves the repository can host audio models with mel features, tokenization, decoder state, GPU acceleration, and VTT/audio-oriented commands.
- `models/speaker` already has ECAPA/SpeechBrain-compatible speaker embedding and fbank/mel extraction. This is relevant for Base model x-vector mode, but should not block CustomVoice.
- `loader/audio` and the speaker/whisper command work provide WAV/non-WAV loading, resampling expectations, and test patterns for audio fixtures.
- `backends/nvidia/runtime` has the low-level pieces needed for GPU talker/code-predictor matmul, LM-head, argmax/sampling, and fused kernels.
- `backends/simd/runtime` and `backends/simd/kernels` provide CPU reference and portable fallback ownership for matmul/RMSNorm/activation/attention primitives.

### Missing or insufficient

- `model/qwen3tts` now covers TTS-specific config parsing, token constants, tensor-group inventory, and deterministic CustomVoice prefix IDs; checkpoint binding and pipeline state are still pending.
- No native Qwen3-TTS talker wrapper that can mix text projections, codec embeddings, MRoPE, speaker/control prefixes, and semantic-token generation.
- No native TTS code predictor package with 15 acoustic heads and compact per-frame KV-cache reuse.
- No `Decoder12Hz` / speech-tokenizer decoder or encoder implementation.
- No Qwen3-TTS CLI, synthesis options, streaming interface, WAV writer integration, or audio fixture suite.
- Existing speaker embedding code may not be weight-compatible with Qwen3-TTS Base checkpoints without explicit tensor-name and preprocessing parity work.

## Proposed package layout

Keep TTS model code separate from the existing text-generation `model/qwen` package while reusing helpers deliberately:

```text
model/qwen3tts/
  config.go          # config.json parser, variant detection, dimensions
  tokens.go          # ChatML/TTS/codec/language/speaker constants
  weights.go         # safetensors binding and required tensor groups
  talker.go          # semantic-token transformer wrapper
  code_predictor.go  # 15 acoustic-group autoregressive decoder
  decoder12hz.go     # codec-frame to waveform decoder
  encoder12hz.go     # optional Base/ICL reference-audio codec encoder
  synth.go           # high-level non-streaming pipeline
  streaming.go       # incremental synthesis/session state
  fixtures_test.go   # reference parity fixtures, shape/tensor checks
```

CLI and scripts:

```text
cmd/qwen3ttsinspect/  # metadata/config/tensor inventory first
cmd/qwen3tts/         # synthesize text to WAV once runtime path exists
scripts/qwen3tts_*    # optional Python/Rust-reference fixture generation helpers
```

If shared Qwen decoder code is extracted, prefer a small internal/common bridge with explicit shape contracts rather than importing all of `model/qwen` into `model/qwen3tts`.

## Implementation roadmap

### Phase T0 — Checkpoint and reference inventory

Goal: make the work measurable before implementing inference.

- [x] Add `docs/qwen3-tts-support.md` and keep it current.
- [x] Add `cmd/qwen3ttsinspect` to read local checkpoint metadata:
  - variant (`base`, `custom_voice`, `voice_design`);
  - size class (`0.6B`, `1.7B`);
  - talker dimensions;
  - code predictor dimensions;
  - speech tokenizer/decoder tensor groups;
  - speaker encoder presence.
- [ ] Add a fixture generator or reference-run notes against `qwen3-tts-rs` for:
  - tokenized prompt/prefix IDs;
  - one talker prefill hidden/logit checksum;
  - first semantic token for a fixed seed;
  - one code-predictor acoustic frame;
  - one short decoded WAV summary/hash.

Acceptance:

- Inspector can identify all five official variants from `config.json` and tensor names without loading full inference state.
- Reference fixture files are small and deterministic enough to commit under `testdata/`.

### Phase T1 — Config, tokenizer, and prefix builder

Goal: reproduce the exact text/control inputs for `0.6B CustomVoice`.

- [x] Implement `model/qwen3tts.Config` and `ParsedModelConfig` equivalent.
- [x] Add token constants and speaker/language enums.
- [x] Reuse `loader/tokenizer` for Qwen tokenizer files; support `tokenizer.json` and fallback `vocab.json` + `merges.txt` if needed.
- [x] Implement CustomVoice prompt builder:
  - ChatML role prefix;
  - TTS pad/BOS/control tokens;
  - language token;
  - speaker token;
  - codec BOS.
- [x] Add table tests for speaker IDs, language IDs, malformed combinations, and known tokenized prefixes.

Acceptance:

- Go prefix/token IDs match `qwen3-tts-rs` fixtures for Ryan/English and at least one non-English speaker.

### Phase T2 — Talker CPU reference

Goal: generate semantic tokens from text on CPU for `0.6B CustomVoice`.

- [ ] Bind talker weights and validate tensor shapes.
- [ ] Implement text embedding, text projection/SwiGLU, codec embedding, MRoPE, decoder layers, RMSNorm, and codec LM head.
- [ ] Reuse checked SIMD runtime APIs for matmul/RMSNorm/activation and maintain scalar/reference tests.
- [ ] Add token suppression for codec control range `[vocab_size-1024, vocab_size)` except EOS.
- [ ] Add deterministic greedy/seeded sampling parity against reference fixtures.

Acceptance:

- First semantic token and a short semantic-token sequence match reference for fixed inputs.
- Malformed dimension/buffer tests cover all exported talker entrypoints.

### Phase T3 — Code predictor CPU reference

Goal: convert semantic tokens and talker hidden state into 16-codebook frames.

- [ ] Bind 15 codec embeddings and 15 acoustic LM heads.
- [ ] Implement 5-layer predictor with prefill on `[talker_hidden, semantic_embed]` and sequential acoustic-group generation.
- [ ] Preallocate per-layer KV cache for the fixed short predictor sequence.
- [ ] Handle 1.7B projection (`small_to_mtp_projection`) behind shape checks, even if first target is 0.6B.
- [ ] Add frame-level tests for one semantic token and one short generated sequence.

Acceptance:

- Acoustic codes for a fixed semantic token match reference fixtures.
- Per-frame predictor path does not allocate growing KV tensors in the hot loop.

### Phase T4 — Decoder12Hz and WAV output

Goal: produce audio for `0.6B CustomVoice`.

- [ ] Inventory decoder/speech-tokenizer tensor names and map ConvNeXt/transposed-convolution blocks.
- [ ] Implement a CPU F32 reference decoder first.
- [ ] Add PCM16 WAV writing or reuse existing audio output helpers.
- [ ] Add numerical summaries for decoded waveform fixtures: sample rate, duration, min/max/RMS, short hash, optional spectrogram hash.

Acceptance:

- A short text prompt writes a valid 24 kHz mono WAV.
- Waveform summary matches reference within agreed tolerance.

### Phase T5 — NVIDIA acceleration and streaming

Goal: make synthesis practical locally.

- [ ] Move talker/code-predictor hot matmuls and LM heads to existing NVIDIA runtime paths.
- [ ] Add GPU-side token suppression and sampling for codec logits.
- [ ] Keep generated code frames on device and defer host transfer until decode or final report.
- [ ] Evaluate whether decoder12hz conv/transposed-conv should use existing CUDA/NVIDIA primitives, CPU first, or a separate kernel set.
- [ ] Add streaming API and CLI mode that yields codec/audio chunks incrementally.

Acceptance:

- `0.6B CustomVoice` generates faster than real time on the local NVIDIA target or has a documented bottleneck profile.
- CPU/GPU semantic/acoustic tokens match for greedy generation.

### Phase T6 — Base and VoiceDesign variants

Goal: cover all official conditioning modes.

- [ ] Add Base x-vector mode using `models/speaker` only after confirming preprocessing/tensor compatibility.
- [ ] Add Base ICL mode by implementing/using `Encoder12Hz` to encode reference audio to codec frames.
- [ ] Add VoiceDesign instruction prompt conditioning.
- [ ] Add validation that rejects or warns for invalid conditioning/model combinations.

Acceptance:

- Base, CustomVoice, and VoiceDesign paths each have one deterministic smoke fixture.
- CLI reports model capability and conditioning mismatches clearly.

## Validation plan

- Use small fixture-first tests; avoid committing model weights.
- Keep model payloads under ignored `models/` directories.
- Run normal gates with workspace temp:

```sh
GOTMPDIR=$PWD/.gotmp go test ./... -run '^$'
GOTMPDIR=$PWD/.gotmp go vet ./...
```

- Add hardware-gated tests for NVIDIA parity once GPU execution lands.
- Add audio-specific checks: WAV header validity, sample rate, duration bounds, RMS/non-silence, and fixture hashes.

## Immediate next action

Implement Phase T0/T1 together: config/token/prefix support plus `cmd/qwen3ttsinspect`. That gives us a safe, testable landing zone before any heavy decoder or codec work.
