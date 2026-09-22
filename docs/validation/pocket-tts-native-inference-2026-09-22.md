# Pocket TTS native inference validation — 22 September 2026

The native Go/SIMD Pocket TTS inference path matches the pinned public English model fixtures and runs faster than real time with zero warm-path heap allocations on the measured amd64 host.

## Inputs

- Source: `kyutai-labs/pocket-tts@0acce6b2f390150267557770d2098c5caa9a18ac`.
- Public no-voice-cloning English model: revision `e7205b6ee50e654a5ea19f0e9df2b0813b05e921`, 219,029,196 bytes, SHA-256 `916ccd2686e9311cb40054893a3c4284393d658825ffc714a276f3e9b152344f`.
- Tokenizer: revision `00eac05ed3d16bdc3f6b5d598874019c34a89214`, 245,020 bytes, SHA-256 `f498428e1eafee50492f7be13dc9bfafcfc12e508cd0eb1b01c92ecd5d8c6687`.
- Alba voice state: revision `e81d79e8194ad4c7ce879c87a4258ef20cbf2487`, 6,194,424 bytes, SHA-256 `69c32db63ca56843d994f81f343f62e0bf2d73f7e4c9bc73e44bb1110b1d8845`.
- Host: Linux/amd64, Intel Core i7-12700, Go 1.26.3, six available Go CPUs, NVIDIA disabled.
- Numerics: source BF16 linear weights with F32 activations; F32 convolution weights; one LSD step; fixed caller-supplied noise.

Released tests require all three `GO_PHERENCE_POCKETTTS_*` environment variables. Each test verifies artifact size and SHA-256 before loading. The default test suite skips these tests without local artifacts and does not use the network.

## Numerical gates

The independent fixtures were produced with upstream Pocket TTS and PyTorch 2.13:

- tokenizer IDs cover ordinary English, repeated spaces, newline, accents and UTF-8 byte fallback;
- tensor inventory verifies 214 BF16 tensors and exact released topology;
- FlowLM final hidden row and EOS logit tolerance: `3e-5`;
- stateless and request-owned streaming FlowLM agree within `3e-5`;
- flow-head output tolerance: `2e-4`;
- imported Alba K/V cache, text prompt, BOS hidden, EOS, fixed-noise velocity and first latent match their pinned values;
- one-frame Mimi and end-to-end waveforms contain exactly 1,920 samples and use tolerance `3e-5`;
- batched and frame-at-a-time Mimi paths agree within `3e-5`;
- PCM16 WAV writing rejects overwrite, non-finite and out-of-range samples and removes failed partial output.

No tolerance was widened during optimisation.

## Allocation-first optimisation

The initial warm benchmark mixed voice preparation with generation and used allocating transformer, flow-head and Mimi helpers. After separating phases, the representative five-frame workload measured:

| Stage | Five frames / 400 ms audio | B/op | allocs/op |
|---|---:|---:|---:|
| Initial mixed path | 952 ms | 75.6 MB | 6,217 |
| Voice preparation separated | 434 ms | 62.0 MB | 6,080 |
| Transformer `StepInto` | 330 ms | 53.4 MB | 746 |
| Mimi convolution `Into` | 381 ms | 7.1 MB | 403 |
| Session-owned FlowHead/Mimi state | 309 ms | 0 | 0 |
| Packed large-column convolution | about 235 ms median | 0 | 0 |
| BF16-resident linear weights | about 181 ms median | 0 | 0 |
| 16-row Mimi transformer chunks | 152–157 ms | 0 | 0 |
| Prepacked transposed-convolution phases | 143–147 ms | 0 | 0 |

A released-model `testing.AllocsPerRun(10, ...)` gate requires exactly zero internal warm allocations.

## Final scaling

Command:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 \
GO_PHERENCE_POCKETTTS_MODEL=/workspace/checkpoints/pocket-tts-without-voice-cloning/languages/english/model.safetensors \
GO_PHERENCE_POCKETTTS_TOKENIZER=/workspace/checkpoints/pocket-tts-without-voice-cloning/languages/english/tokenizer.json \
GO_PHERENCE_POCKETTTS_VOICE=/workspace/checkpoints/pocket-tts-without-voice-cloning/languages/english/embeddings/alba.safetensors \
go test ./model/pockettts -run '^$' \
  -bench 'BenchmarkReleased(FirstChunk|FiveFrames|TwentyFiveFrames)$' \
  -benchtime=3x -benchmem -count=5
```

| Workload | Audio duration | Measured range | Real-time factor | B/op | allocs/op |
|---|---:|---:|---:|---:|---:|
| First chunk | 80 ms | 51–54 ms | 1.48–1.57× | 0 | 0 |
| Five frames | 400 ms | 143–147 ms | 2.72–2.80× | 0 | 0 |
| Twenty-five frames | 2.0 s | 398–427 ms | 4.68–5.02× | 0 | 0 |

Cold CLI execution was measured separately with a prebuilt binary. The initial path needed 1.30 seconds and about 589 MB maximum RSS for model load, voice load, five generated frames and WAV output. The final BF16-resident/session path takes 0.41 seconds and about 450 MB maximum RSS for the same workload. Warm generation results do not include model/tokenizer/voice loading.

## Profile outcome

The initial CPU Sankey is attached to the project conversation as `pocket-tts-cpu-profile.svg`. Repeated profiles after each material change moved the dominant cost from allocation/layout churn and naïve convolution to real matrix work. The final retained serial path uses:

- BF16-resident linear weights with repository BF16/F32 SIMD GEMV;
- request-owned K/V, transformer, FlowHead and Mimi scratch;
- zero-allocation `StepInto`, `DecodeFrameInto` and `GenerateInto` APIs;
- 16-row batched Mimi transformer projections;
- caller-scratch packed SGEMM for large-column causal convolutions;
- prepacked transposed-convolution phase matrices.

A persistent two-worker BF16 GEMV pool was tested but removed: its first-chunk results were noisy and its long-run gain was smaller than the accepted batching/prepacking changes.

## Limits

- Raw-audio voice cloning needs the gated full model bundle and a pinned prompt fixture. The public path imports precomputed voice states.
- The CLI currently processes one prepared text chunk per invocation. Long-text sentence chunking can be added without changing model arithmetic.
- Native ARM64 and RVV execution benchmarks have not run. Cross-builds only establish source compatibility.
- Training is a separate work item. No backward, optimizer or training-checkpoint claim follows from these inference results.
