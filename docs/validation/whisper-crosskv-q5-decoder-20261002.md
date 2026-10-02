# GPU cross K/V and fused Q5_0 CPU decoder — 2 October 2026

Two explicit opt-ins on the private `vulkan-original-q5-padded-integer-dot-mmq-tanh` backend, combined with `OriginalDecoderCompatibility` and `OriginalWindowCompatibility`, cut the JFK request from 4.456 s to 4.064 s. On the podcast-vadwords fixture the request goes from 8.219 s to 5.871 s. Transcripts, timestamps and word timings are unchanged on all 10 fixtures. Defaults, services and resources are unchanged. Pinned whisper.cpp is still faster on JFK (about 3.02 s).

## Stage attribution

New benchmark timers split each request into mel, encoder, decoder-state construction and decoder phases (`MelSeconds`, `EncodeSeconds`, `DecoderStateSeconds`). JFK medians, before this change, compared with whisper.cpp's own timers:

| Stage | Go (MMQ-tanh + compat) | whisper.cpp |
|---|---:|---:|
| mel | 0.09 s | ~0.01 s |
| encoder | 3.39 s | 2.78 s (encode, including conv and cross K/V) |
| cross K/V state (CPU F32) | 0.20 s | inside encode, 0.023 s on GPU |
| decoder loop | 0.65 s | ~0.2 s |

The fenced per-operator encoder profile (`stageprof-mmqtanh.json`) attributes about 1.67 s to the projections and Q8_1 quantisation, against 1.03 s in the original GPU-timed profile, and 1.76 s to attention, against 1.58 s for the original's flash attention. On Intel without cooperative matrices the original also uses the 64×64 medium integer-dot tile (`mul_mat_l = coopmat_support`), so tile choice does not explain the projection gap. One remaining structural difference is that the original reads Q8_1 activations as `block_q8_1_x4` with 128-bit loads.

## GPU cross K/V (`GO_PHERENCE_WHISPER_BENCH_GPU_CROSS=1`)

`newVulkanEncoderOriginalCross` appends each decoder layer's cross-attention key and value projections to the encoder's final plan. Each is an original-format Q5_0×Q8_1 MMQ of the final hidden state, as in whisper.cpp's encode graph. One Q8_1 quantisation of `h` is shared by all eight projections. Q5 weights come from the pinned file and biases from the decoder. The encoder downloads the K/V rows, and `newDecoderStateFromCrossContext` adopts them.

This changes the cross K/V arithmetic from F32 to the original's Q5×Q8_1. On all 10 fixtures × 5 repeats, outputs were byte-identical to the CPU F32 cross K/V arm. Decoder-state construction fell from about 0.20 s to 0.014 s per window, and requests got 1.8–3.2% faster.

## Fused Q5_0 CPU decoder (`GO_PHERENCE_WHISPER_BENCH_DECODER_Q5=1`)

The CPU decoder streamed about 580 MB of F32 weights per token, which made it memory-bound. `simd.SdotQ5_0` (AVX2/FMA/F16C, `backends/simd/runtime/sdot_q5_0_amd64.s`) dequantises each Q5_0 value in registers with the loader's single-rounding expression, `f16(d) * float32(q-16)`. It then accumulates in exactly `Sdot`'s order: alternating 8-lane accumulators, then the same horizontal reduction.

`Decoder.attachOriginalQ5` loads the file's packed rows for every decoder projection and the tied LM head. It checks every packed value bit-equal to the F32 weight it replaces, and rejects CPUs without the kernel and the int8/GPU decoder paths. Results are bit-identical to the F32 decoder, while weight traffic is about 5.8× lower.

Tests:
- 1,000 random rows: bit-identical to `Sdot` over the widened row, including subnormal activations, extreme F16 scales and signed zeros.
- Synthetic two-layer decoder: logits bit-identical over 5 tokens, with and without decoder compatibility, at 1 and 4 workers.
- `linearQ5Into` matches `linearInto` bit for bit on production shapes.

## Results (5 repeats, medians; every repeat's output identical)

| Fixture | Baseline (CPU cross K/V) | + GPU cross K/V | + Q5 decoder | Output |
|---|---:|---:|---:|---|
| JFK | 4.456 | 4.354 / 4.324 | **4.064** | same, 0–10.40 |
| PT row 0 | 4.466 | 4.322 / 4.307 | **4.054** | same, 0–7.36 |
| FR row 0 | 4.093 | 3.964 / 3.940 | **3.842** | same |
| JFK VAD+words | 5.186 | 5.046 / 5.040 | **4.439** | same |
| groups | 10.512 | 10.230 / 10.239 | **8.970** | same |
| PT2 | 10.441 | 10.198 / 10.185 | **8.912** | same |
| podcast | 6.111 | 5.994 / 6.010 | **4.854** | same |
| podcast VAD+words | 8.371 | 8.218 / 8.219 | **5.871** | same |
| PT1 | 4.118 | 4.006 / 4.016 | **3.851** | same |
| silence | 0.021 | 0.021 / 0.021 | 0.021 | same |

The GPU cross K/V column shows two separate runs. The Q5 column ran interleaved with the second of them.

JFK decoder phases (self / cross / MLP / head): 0.104 / 0.213 / 0.168 / 0.169 s became 0.047 / 0.153 / 0.087 / 0.084 s. The remaining cross phase is mostly CPU cross-attention over 1,500 keys.

## Follow-up: parallel cross-attention heads and sparse mel

Both changes are exact:

- **Parallel cross-attention heads.** The attached fast decoder splits unobserved cross-attention heads across workers. Each head keeps its serial arithmetic and writes only its own output slice. Alignment-observed attention stays serial.
- **Sparse mel filterbank.** The filterbank sums only each mel's non-zero bin span. The power spectrum is finite and non-negative, and adding `+0·power` to a sum that starts at +0 never changes it. Retained terms keep ascending bin order. Frames are split across up to 4 workers.

Mel output and maximum are bit-identical to the dense serial loop for 80 and 128 bands, window lengths 160–480,000 samples, at 1, 3 and 4 workers. Parallel heads are bit-identical at 2, 3, 4 and 7 workers, with 0 and 36 padded keys. Mel time per 30 s window fell from 93 ms to 12.5 ms.

| Fixture | Q5 decoder | **+ parallel heads, sparse mel** | Output |
|---|---:|---:|---|
| JFK | 4.064 | **3.825** | same |
| PT row 0 | 4.054 | **3.784** | same |
| FR row 0 | 3.842 | **3.662** | same |
| JFK VAD+words | 4.439 | **4.153** | same |
| groups | 8.970 | **8.344** | same |
| PT2 | 8.912 | **8.118** | same |
| podcast | 4.854 | **4.410** | same |
| podcast VAD+words | 5.871 | **5.462** | same |
| PT1 | 3.851 | **3.690** | same |
| silence | 0.021 | 0.021 | same |

JFK stages (median request) are now:

| Stage | Time |
|---|---:|
| mel | 0.012 s |
| encoder | 3.529 s |
| decoder state | 0.021 s |
| decoder: self / cross / MLP / head | 0.038 / 0.062 / 0.066 / 0.068 s |

## Remaining gap (JFK 3.83 s vs 3.02 s)

- Encoder: 3.53 s vs 2.78 s. This is now almost the whole gap. Targets are projections (Q8_1 x4 layout / 128-bit loads) and attention (flash attention, F16 K/V).
- Decoder: 0.23 s vs ~0.2 s. Prompt tokens are still decoded one at a time, which matters for multi-window clips.

## Evidence

`benchmarks/speech-foundations/whisper-crosskv-q5decoder-20261002/`:
- `stagesplit-*`: stage split;
- `stageprof-mmqtanh.json`: fenced operator profile;
- `xkvbase-*`, `xkv-*`: GPU cross K/V arms;
- `q5ref-*`, `q5dec-*`: Q5 decoder arms;
- `fast2-*`: parallel-heads / sparse-mel arm;
- driver scripts and logs, plus `SHA256SUMS`.

All runs used the isolated window: CPU4, 8 GiB, no swap, physical Intel Iris Xe, Qwen idle. The first cross-K/V drive was interrupted after 4 runs (logged), and the remaining 16 ran in a fresh window.
