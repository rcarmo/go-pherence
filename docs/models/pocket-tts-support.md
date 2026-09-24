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
- Native frozen-Mimi raw-audio encoding covers the SEANet encoder, projected causal transformer and replicate-padded 16× downsampling. The pinned 30,721-sample fixture produces 17×32 latents within `3e-5` of upstream and loads through the frozen latent-cache boundary. `EncodeInto` uses bounded 16-frame chunks, rejects overlapping input, output and workspace storage before mutation, and performs zero warm heap allocations. On an Intel i7-12700, a 30-second input took 4.29–4.77 seconds after setup; the 375-frame workspace allocated 91,912,120 bytes in 66 allocations.
- Autoregressive generation orchestration with caller-owned noise, EOS countdown, persistent FlowLM/Mimi state, exact 1,920-sample chunks and exclusive PCM16 mono WAV output. A released one-frame fixture validates voice-state → text → FlowLM → LSD → Mimi → PCM as one contract.
- Session-owned `StepInto`, `DecodeFrameInto` and `GenerateInto` paths perform zero warm heap allocations. The measured Intel i7-12700 path emits the first 80 ms chunk in about 51–54 ms, 400 ms of audio in 143–147 ms, and 2 seconds in 398–427 ms. See [the validation record](../validation/pocket-tts-native-inference-2026-09-22.md).
- `cmd/audio/pockettts` loads external config, model, tokenizer and voice-state files and writes a seeded WAV without Torch, ONNX Runtime, CGo model runtimes or Python subprocesses.

## Next inference slices

1. Connect the released raw-audio Mimi encoder to a user-facing voice-cloning flow with prompt validation and end-to-end generated-audio parity.
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
- a direct pinned `TrainableTTS.forward` fixture with shared sampled noise matches raw/normalized flow metrics, EOS, total loss, both log-variance leaf gradients, audio/voice inputs and every FlowLM/flow-head/`w_s_t` parameter within `8e-5`;
- full-model AdamW/EMA matches PyTorch after one step; versioned directory-durable checkpoints include trainables, optimizer moments/settings, EMA, mutable latent statistics and fixed timestep frequencies, and resumed second-step state is byte-for-byte equivalent to uninterrupted state;
- upstream frozen-Mimi latent stores are admitted with exact sidecar fields, Mimi SHA-256 binding, index-derived shard names, root confinement, bounded F32 `[frames,32]` tensors, per-shard immutable content digests and non-finite rejection; native deterministic F32 shard output loads through upstream `safetensors`, and fixed padded overlap stitching preserves upstream valid-mask semantics;
- allocation-free production-shape admission covers the upstream 30 s target (375 frames), 5 s prompt (62 frames), explicit text/sequence/resident-byte ceilings and effective batches through gradient accumulation. The released worst-case 512-token row is 950 sequence rows, with exactly 89,449,730 trainables and a conservative ≤5.35 GiB resident bound including parameters, gradients, Adam moments, EMA, current F32 tapes and one garbage-collector cycle of dead tapes. The 24-layer 316,015,874-parameter teacher is also admitted under an explicit 64 GiB ceiling. Live topology/storage is revalidated before workspace allocation;
- deterministic released-format export matches upstream state semantics: raw FlowLM parameters and live buffers, partial EMA overlay for tracked parameters, one FlowMatching or two LSD time conditions, no training-only `w_s_t`, and the exact frozen Mimi module state. A pinned independent fixture freezes all 87 released Mimi names/shapes; source BF16 is widened through F32 as upstream's live Mimi module does. Lexical safetensors output is atomically published with file/directory sync and typed post-publication sync uncertainty;
- exact depth/CFG distillation computes frozen teacher full/null passes and `z_null + cfg*(z_cond-z_null)`, then applies hidden-dimension MSE over the upstream shifted mask. Direct pinned PyTorch parity covers loss, student/teacher/target rows, audio/voice inputs and every active student gradient. The champion ends seed maps 24 layers to `[0,1,2,21,22,23]`; a dedicated trainer registers only conditioning/backbone parameters, so EOS, flow head and `w_s_t` cannot move under AdamW decay or EMA;
- a deterministic tiny-graph CPU soak executes 100,000 uninterrupted and 100,000 periodic-resume complete steps with step-varying explicit samples. State matches at every 10,000-step boundary and final logical/checkpoint hashes repeat across two runs; post-GC retained heap remains bounded. This is checkpoint/resume qualification for the exact tiny graph, not released-shape performance evidence.

See [the training validation record](../validation/pocket-tts-native-training-2026-09-22.md) for the fixture and commands.

Released-topology profiling now uses mutable F32 weights loaded directly from the pinned checkpoint, the exact zero-initialised default normalized-LSD weighting network, 375 target frames, 62 prompt frames and 32 deterministic text tokens. On an Intel i7-12700, checked SIMD and batched transformer linears reduced one forward/backward from 104.19 seconds to 22.94 seconds on one core and 19.65 seconds with six Go CPUs. Warm AdamW plus EMA over 89,449,730 parameters took 594.6 ms with zero allocations. The forward/backward still allocates 1.56 GB in about 353,000 objects because per-frame primal/dual flow tapes are not arena-backed.

A 100-update CPU soak passed on the released six-layer topology with a deterministic 470-row synthetic workload in 33m29.856s. The ten-step post-GC live heap series stayed within about 46 KiB; peak process RSS was 4,328,756 KiB. This exercises finite full-model AdamW/EMA updates and bounded live heap, not training quality, released-shape checkpoint/resume, the 512-token worst case or the 24-layer teacher.

A bounded request-owned per-frame primal/dual flow-tape scratch now preserves exact released-row owned-tape metrics and every gradient. Ten complete updates pass with no scratch spill. One warm 470-row forward/backward allocates 1.254 GB in 220,726 objects, down 19.7% in bytes and 37.5% in objects from the pre-scratch path; three timing samples overlap, so latency has no accepted gain. The earlier 100-update gate ran before this scratch change. A separate bounded primary dual-backward scratch reduces the warm row again to 1.094 GB and 151,351 objects; setup rises by 9,520 bytes and two allocations, and no latency gain is established. Released one-step exact parity and no-spill gates pass; the ten-update tape soak predates this third pool. A fourth bounded pool reuses ordinary/endpoint backward temporaries. Exact released all-gradient parity, zero spills and separate ten- and 100-update CPU soaks pass; the warm 470-row step allocates 946.3 MB in 88,351 objects. The four-pool 100-update synthetic soak passed in 33m47.172s at `55c2f7fc`, final loss `0.006236044`, peak RSS 4,304,368 KiB; it does not establish trained-speech quality. Released-shape in-memory resume and two successive 1.431 GB binary checkpoint save/load boundaries match continuous training through the third update exactly. The legacy JSON reader's 512 MiB limit still precludes a released JSON checkpoint. File-backed resume peaked at 13.15 million KiB RSS during a concurrent soak and 10.41 million KiB on an idle host in the two-boundary test, outside ordinary training admission; reserve storage and memory separately. A subsequent layer-local transformer probability-gradient row reduces the warm step to 853.9 MB / 43,237 allocations. Exact released all-gradient parity and a separate ten-update synthetic soak pass; the 100-update result predates this change. A further layer-local norm backward reuse passes released exact-gradient parity, zero-affine-weight central differences and a separate ten-update synthetic soak; its warm 470-row step allocates 779.0 MB / 24,933 objects. The 100-update result predates both transformer changes. Workspace-local LSD row-result scratch passes released exact-gradient parity and a separate ten-update no-spill soak; the warm step now allocates 776.6 MB / 18,183 objects. The standalone API still returns owned slices, and the 100-update run predates this change. A further transformer norm forward row reuse passes released exact-gradient parity, a ten-update soak and an independent 100-update no-spill soak at `fbb65719`; the warm step allocates 751.6 MB / 12,086 objects. The latest synthetic 100-update run took 32m9.151s with final loss `0.006236044` and peak RSS 4,246,144 KiB. An additional transformer tape-output copy removal passes exact released gradients, owned-output tests, and separate ten- and 100-update soaks; its warm row allocates 715.1 MB / 12,067 objects. The 100-update synthetic run at `ecd5dc1b` passed in 33m30.002s with final loss `0.006236044`, finite state, zero spills and peak RSS 4,299,008 KiB. Opt-in ten- and 100-update 512-token, 950-sequence-row synthetic soaks pass with finite state and zero spills. The pinned 100-update run at `f7544fdc` ended with loss `0.007436165`, peak RSS 5,333,980 KiB, and bounded post-GC heap. Trained-speech quality remains unqualified. Other row outputs and retained memory need a fresh profile. GPU training is a separate gate.

Preset-voice inference and the reusable released raw-audio Mimi encoder are complete for the pinned artefacts. A user-facing voice-cloning flow still needs prompt validation and end-to-end generated-audio parity. Full native training is incomplete.
