# Qwen3-TTS support roadmap

This note maps <https://github.com/TrevorS/qwen3-tts-rs> and the official Qwen3-TTS checkpoints onto the current `go-pherence` repository, then lays out a staged native implementation plan.

## Source reviewed

- Reference implementation: <https://github.com/TrevorS/qwen3-tts-rs/tree/711ceee07cad92673f86de8997bdf54c30caa49f> (`711ceee07cad92673f86de8997bdf54c30caa49f`)
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
- `model/whisper` audio frontend/decoder infrastructure and GPU LM-head work;
- `model/speaker` ECAPA/SpeechBrain speaker embedding, fbank/mel extraction, smoothing, and validation commands;
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
- `model/whisper` proves the repository can host audio models with mel features, tokenization, decoder state, GPU acceleration, and VTT/audio-oriented commands.
- `model/speaker` already has ECAPA/SpeechBrain-compatible speaker embedding and fbank/mel extraction. This is relevant for Base model x-vector mode, but should not block CustomVoice.
- `loader/audio` and the speaker/whisper command work provide WAV/non-WAV loading, resampling expectations, and test patterns for audio fixtures.
- `backends/nvidia/runtime` has the low-level pieces needed for GPU talker/code-predictor matmul, LM-head, argmax/sampling, and fused kernels.
- `backends/simd/runtime` and `backends/simd/kernels` provide CPU reference and portable fallback ownership for matmul/RMSNorm/activation/attention primitives.

### Missing or insufficient

- `model/qwen3tts` covers TTS-specific config parsing, token constants, tensor-group inventory, deterministic CustomVoice prefix IDs, owned F32 Talker weight binding, CustomVoice prefill, GQA/RoPE execution, token suppression, and greedy first-semantic-token generation.
- Multi-frame Talker continuation uses 15 CodePredictor acoustic embeddings per frame and request-local Talker KV/scratch. The capped greedy CPU reference accepts 1–32 frames; the thirty-two-frame `Hi` seed-42 probe now has pinned released-model parity, while the fixed two-/three-/four-frame APIs retain their exact sizes.
- The native CodePredictor has 15 acoustic heads and allocates request-local KV per frame. Warm per-frame workspace reuse remains open.
- The owned-F32 CPU `Decoder12Hz` binds normalized 16-codebook tables, the eight-layer causal pre-transformer, ConvNeXt and BigVGAN-style transposed-convolution stages. Its 24 kHz mono PCM output has pinned released-tokenizer parity through eight frames; the fixed two-frame command writes a no-clobber PCM16 WAV.
- There is no general-purpose text-to-WAV or streaming interface. The fixed two-frame command is a short parity probe.
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
cmd/qwen/qwen3ttsinspect/  # metadata/config/tensor inventory first
cmd/qwen3tts/         # synthesize text to WAV once runtime path exists
scripts/qwen3tts_*    # optional Python/Rust-reference fixture generation helpers
```

If shared Qwen decoder code is extracted, prefer a small internal/common bridge with explicit shape contracts rather than importing all of `model/qwen` into `model/qwen3tts`.

## Implementation roadmap

### Phase T0 — Checkpoint and reference inventory

Goal: make the work measurable before implementing inference.

- [x] Add `docs/models/qwen3-tts-support.md` and keep it current.
- [x] Add `cmd/qwen/qwen3ttsinspect` to read local checkpoint metadata:
  - variant (`base`, `custom_voice`, `voice_design`);
  - size class (`0.6B`, `1.7B`);
  - talker dimensions;
  - code predictor dimensions;
  - speech tokenizer/decoder tensor groups;
  - speaker encoder presence.
- [ ] Add reference-run values against `qwen3-tts-rs` for:
  - [x] tokenized prompt/prefix IDs fixture schema and deterministic CustomVoice prompt fixture;
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

- [x] Bind dense F32 Talker weights transactionally and validate exact tensor shapes.
- [x] Implement text embedding, biased text projection/SiLU, codec embedding, GQA/RoPE decoder layers, RMSNorm, and codec LM head for CustomVoice prefill.
- [x] Reuse checked SIMD runtime APIs for GEMV, RMSNorm, activation, RoPE, attention, and residual operations.
- [x] Add greedy token suppression for codec control range `[vocab_size-1024, vocab_size)` except EOS.
- [x] Add deterministic synthetic first-token, malformed tensor, forged-control, ownership, and non-finite tests.
- [x] Capture exact first-token Talker hidden/logit rows and semantic token from a pinned 0.6B CustomVoice Rust/Candle CPU oracle. The legacy full-run placeholder remains unqualified.
- [x] Qualify up to fifteen greedy continuations after first-token prefill using 15 acoustic embeddings per frame and request-local Talker KV. The capped eight-frame reference returns semantic IDs `[1995, 215, 212, 1181, 462, 251, 530, 122]`; the same prompt's sixteen-frame reference continues with `[1792, 1792, 1086, 1086, 1086, 1724, 1724, 1792]`. All 240 acoustic codes match Rust/Candle. It exercises the projected trailing text token, TTS EOS text embedding, then TTS PAD. EOS early stopping has synthetic coverage; a separately named `GenerateCappedGreedyMinTwoCPU` withholds EOS until two semantic tokens have been selected, matching the pinned Rust library's minimum-token policy. The standalone Rust `generate_audio` loop does not apply that minimum. Pinned Rust/Candle CPU sampling probes record top-k ties, CPU top-p ordering, greedy selection, repetition adjustment and reserved-token suppression, plus seeded first-token choices on the released prefill logits. A separately labelled controlled near-tie logit row exercises stochastic outcomes; it is not released-model output. `ReferenceCPUSampler` also drives a separately named `GenerateCappedSeededCPU` probe (seed 42, Rust CPU defaults). Pinned sixteen-frame Rust/Candle and Go runs agree on all semantic and acoustic IDs for seeds 42 and 7. Seed 42's 30,720 samples differ by at most `5.90459e-6` under its `6.5e-6` limit; seed 7 follows a different semantic path and differs by `3.03984e-6` under its separate `3.5e-6` limit. The earlier greedy fixtures and limits remain unchanged. The short one-token prompt `Hi` (token `13048`) also matches seed-42 Rust/Candle across sixteen semantic and 240 acoustic IDs. Its 30,720 samples differ by at most `1.34856e-6` under a separate `1.6e-6` limit. Both `Hello world` seeds and `Hi` fill the 16-frame Go cap without EOS. A `Hi` seed-42 probe filled a thirty-two-frame cap without EOS in both Rust/Candle and Go. All 512 frame-major IDs match; the 61,440-sample waveform differs by at most `1.34856e-6` under the existing `1.6e-6` `Hi` limit. Go's capped CPU reference permits up to 32 frames. One two-request `Hi`/seed-42 concurrency admission run shared the pinned weights, matched both oracle outputs, and stayed below 6 GiB peak RSS without post-GC heap growth; its race-instrumented counterpart reported no data race. A separate two-request `Hi`/seed-42 plus `Hello world`/seed-7 sixteen-frame run also matched both independent fixtures with owned results and no race. Three successive pairs of 32-frame `Hi` requests also retained six independent outputs while later pairs ran; post-release GC restored the baseline heap and observed peak RSS stayed below 6 GiB. More workers, hours-long retention, general-length generation and EOS behaviour lack admission evidence. Other seeds, longer autoregression, natural EOS, intelligibility and general-purpose stochastic synthesis still need qualification.

Acceptance:

- The CPU path produces a deterministic first semantic token from the exact ten-position CustomVoice prefill.
- The pinned first-token fixture passes with max absolute error `4.14849e-5` (1024 hidden values) and `2.50340e-5` (3072 raw logits), compared with fixed limits `5e-5` and `3e-5`. Both select semantic token `1995`; bytewise hashes differ because CPU implementations round differently. Reproduce with `GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=<pinned-model-dir> GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^TestTalkerReleasedPrefill$' -count=1 -v`. With no model directory the test skips; a skipped gate is not a pass. Fixture and oracle source are under `model/qwen3tts/testdata/customvoice_0b6_ryan_hello/` and `scripts/qwen3tts_oracle_prefill.rs`.
- Malformed dimensions, prompt controls, buffers, and non-finite logits fail closed.

### Phase T3 — Code predictor CPU reference

Goal: convert semantic tokens and talker hidden state into 16-codebook frames.

- [x] Bind 15 codec embeddings and 15 acoustic LM heads for equal-width 0.6B CustomVoice.
- [x] Implement a scalar Go/SIMD 5-layer first-frame predictor with two-position prefill and 14 sequential acoustic steps.
- [x] Reuse bounded per-layer KV and scratch across frames within one capped request. The first-frame API allocates its own request-local workspace; returned codes and first logits are owned. There is no shared global cache.
- [ ] Implement 1.7B projection (`small_to_mtp_projection`); the CPU loader rejects the unimplemented geometry.
- [x] Test one synthetic first frame and the released first 15-code frame against pinned Rust/Candle CPU. Bounded two-, three- and four-frame references pin subsequent acoustic frames and waveforms.

Acceptance:

- The pinned first frame for Ryan/English `Hello world` produces acoustic codes `[1782, 1114, 1459, 1531, 804, 506, 240, 1440, 86, 743, 1382, 914, 781, 82, 803]`. The raw first-head 2048-logit row differs by at most `4.00544e-5`, within the fixed `5e-5` threshold. The opt-in released test hashes the model before loading it; with no model directory it skips. See `model/qwen3tts/testdata/customvoice_0b6_ryan_hello/` and `scripts/qwen3tts_oracle_acoustic.rs`.
- Bounded two- and three-frame Talker continuation match the pinned Rust/Candle reference. Second hidden and raw logits maximum absolute errors are `6.19889e-5` and `3.67165e-5` under fixed `7e-5` and `4e-5` limits. The third semantic token is `212`; all 45 acoustic IDs agree. Third hidden and raw logits maximum absolute errors are `1.25886e-4` and `4.02928e-5` under fixed `1.4e-4` and `5e-5` limits. The fourth semantic token is `1181`; all 60 acoustic IDs agree. Fourth hidden/logits maximum absolute errors are `9.15528e-5`/`3.24548e-5` below fixed `1e-4`/`4e-5` limits. The capped greedy eight- and sixteen-frame tests check all frame-major code IDs against the pinned `eight_frame_codes.u32le` and `sixteen_frame_codes.u32le` fixtures. Talker KV and layer scratch are request-local; the CodePredictor resets KV lengths within a request-local workspace while retaining bounded capacity across frames. The released first-head logits and frame codes remain owned.

### Phase T4 — Decoder12Hz and WAV output

Goal: produce audio for `0.6B CustomVoice`.

- [x] Inventory decoder/speech-tokenizer tensor names and map codebooks, causal transformer, ConvNeXt, SnakeBeta, residual, and transposed-convolution blocks.
- [x] Implement an owned-F32 CPU reference decoder with exact causal padding/right trimming and 1920× topology output geometry.
- [x] Add exclusive 24 kHz mono PCM16 WAV writing with finite/range/partial-output checks.
- [x] Pin the released first-frame Decoder12Hz float32 waveform and source hashes for Ryan/English `Hello world`. The tokenizer checkpoint has a 512-wide codebook projection and 512-wide transformer hidden state with 1024-wide attention projections; the older Go default and synthetic pre-upsampling kernel were corrected to accept the released tensors. The independent Rust/Candle reference produces 1,920 samples at 24 kHz, min/max `-0.000239736`/`0.000229278` and RMS `0.0000994396`; Go maximum absolute sample error is `7.26141e-9` under a fixed `1e-8` limit. One 80 ms frame is not speech-quality evidence. No spectrogram has been qualified.

Acceptance:

- The pinned first semantic token and 15 acoustic codes decode to an owned 24 kHz mono 1,920-sample waveform, verified in `model/qwen3tts/testdata/customvoice_0b6_ryan_hello/waveform.f32le`. Set `GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=<pinned-model-dir>` and run `GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^TestDecoder12HzReleasedFirstFrame$' -count=1 -v`. The test hashes both model files before loading; without them it skips.
- `GenerateTwoFramesCPU` now joins the native Talker, CodePredictor and Decoder12Hz for a bounded greedy two-frame request. Its 3,840 F32 samples (160 ms) differ from the pinned Rust/Candle waveform by at most `7.49424e-9` under the fixed `1e-8` limit. Set `GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=<pinned-model-dir>` and run `GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^TestTwoFrameCPUReleased$' -count=1 -v`; the test hashes both released checkpoints first. The fixed `cmd/audio/qwen3tts-twoframe` CPU probe writes this result as a no-clobber 24 kHz PCM16 WAV after hashing both model assets. `GenerateThreeFramesCPU` passes a pinned 5,760-sample (240 ms) Rust/Candle waveform with maximum absolute error `7.49424e-9` below `1e-8`; set `GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=<pinned-model-dir>` and run `GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^TestThreeFrameCPUReleased$' -count=1 -v`. `GenerateFourFramesCPU` also passes a pinned 7,680-sample (320 ms) waveform within `7.49424e-9` of Rust/Candle; set `GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=<pinned-model-dir>` and run `GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^TestFourFrameCPUReleased$' -count=1 -v`. `GenerateCappedGreedyCPU` accepts a cap of 1–32 frames and returns complete frames when EOS arrives before the cap. The released `Hello world` eight-frame oracle produces 15,360 samples (640 ms); the maximum absolute Go/Rust difference is `9.16422e-7` under the frozen `1e-6` threshold. Run `GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^TestCappedGreedyCPUReleasedEightFrames$' -count=1 -v` with the same environment variable; the test verifies fixture, script and checkpoint SHA-256 values. A separate sixteen-frame gate checks 30,720 samples (1.28 s) with maximum absolute difference `1.180917e-6` below its frozen `1.3e-6` limit; run `GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/qwen3tts -run '^TestCappedGreedyCPUReleasedSixteenFrames$' -count=1 -v` with the same environment variable. The eight-frame limit remains `1e-6`. The released prompt has no EOS within sixteen greedy or seed-42 sampled frames. `GenerateCappedGreedyMinTwoCPU` is a separate, synthetic-tested greedy policy. `GenerateCappedSeededCPU` has a sixteen-frame seed-42 CPU parity probe; neither API establishes natural EOS or general seeded sampling behavior. General-purpose text-to-WAV, natural EOS on longer prompts, intelligibility and production throughput remain open.

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

- [ ] Add Base x-vector mode using `model/speaker` only after confirming preprocessing/tensor compatibility.
- [ ] Add Base ICL mode by implementing/using `Encoder12Hz` to encode reference audio to codec frames.
- [ ] Add VoiceDesign instruction prompt conditioning.
- [ ] Add validation that rejects or warns for invalid conditioning/model combinations.

Acceptance:

- Base, CustomVoice, and VoiceDesign paths each have one deterministic smoke fixture.
- CLI reports model capability and conditioning mismatches clearly.

## Validation plan

- Use small fixture-first tests; avoid committing model weights.
- Keep model payloads under ignored `checkpoints/` directories.
- Run normal gates with workspace temp:

```sh
GOTMPDIR=$PWD/.gotmp go test ./... -run '^$'
GOTMPDIR=$PWD/.gotmp go vet ./...
```

- Add hardware-gated tests for NVIDIA parity once GPU execution lands.
- Add audio-specific checks: WAV header validity, sample rate, duration bounds, RMS/non-silence, and fixture hashes.

## Immediate next action

Keep the capped CustomVoice CPU path behind released-checkpoint gates. Profile before each optimisation: the decoder causal convolution dominates CPU time, while request-local Talker and CodePredictor scratch reduce allocation volume without a supported latency improvement. Do not infer intelligible speech or natural EOS from the short greedy fixtures. A separate pinned Rust/Candle `Hi` seed-42 CPU probe, bounded at 64 steps, sampled natural EOS at step 46 after 46 complete acoustic frames; see [the validation record](../validation/qwen3-tts-sixteen-frame-20260923.md). Go remains capped at 32 frames and has not reproduced that EOS stop. The WAV attachments appeared empty on a tablet; same-audio MP3 files played. One listener recognised `Hi` but heard laughter instead of `Hello world` in the capped seed-7 sample. That failure is also present in the pinned Rust waveform, so numerical parity does not establish prompt intelligibility. General intelligibility and voice quality remain unqualified.
