# Whisper and transcription-app optimisation map — 2 October 2026

This map starts from the [final comparison](whisper-final-comparison-20261002.md). Per request, Go is 2.7–7.7% faster than whisper.cpp; the JFK request takes 2.79 s, of which the encoder is 2.52 s and the decoder about 0.21 s. Estimates below come from the [stage profile](whisper-attention-q48-20261002.md), CPU microbenchmarks and harness reports from 2 October. Every item must keep outputs bit-identical unless marked **requalify**.

## 1. Use it in the app (largest user-visible gain)

**Done:** [deployed](transcribe-web-whisper-q5-deploy-20261002.md) on 2 October 2026 (`2dc4a9df`). The packed-only GGML load and the resident model are in place; peak memory is 4.06 GB.

| Item | Why | Size | Blocker |
|---|---|---|---|
| Wire the Q5 Vulkan Whisper path into `speechjobserve` | The app runs CPU Nemotron (JFK job 7.0 s wall, speakers included). Whisper Vulkan ASR is about 2.8 s per 30 s window; Nemotron CPU runs at RTF ≈ 0.6. | ~2–4× faster ASR; word-level speaker labels return (Nemotron currently labels 0 JFK words) | The service needs GPU access: removing `PrivateDevices=yes` from the `nemotron-cpu` drop-in means sharing the Intel GPU with Qwen. Also needs a public constructor (GGML Q5 file → MMQ-tanh encoder + GPU cross + Q5 decoder + compat options; today it is wired only in the test harness), a pinned GGML asset and config/profile schema. |
| Packed-only Q5 load for the GGML path | The loader widens Q5 to F32 (5.0 GB allocated at load), which conflicts with the 8 GiB unit limit | −3 to −4 GB; load 2.9 s → <1 s | Loader change (packed-only source exists for the safetensors path, `eb1582ce`) |
| Resident model and plan | Setup is 2.89 s load + 1.21 s preparation against the original's 0.29 s. With 5 requests including setup, Go loses (18.1 vs 16.7 s) | −4 s per cold job | Resident memory/GPU policy |
| Native Silero VAD in app profiles | Skips silence windows; the compact VAD path is qualified on text | Large on sparse audio | Compact VAD + words fails closed; decide whether the app needs words with VAD |

## 2. Encoder (GPU, 2.52 s per window)

Fenced stage totals: linear 1.22 s (FC1 0.39, FC2 0.40, Q/K/V/O ~0.11 each), attention 1.08, Q8_1 activation quantisation 0.21 (192 dispatches), small ops 0.32.

| # | Item | Estimate | Notes |
|---|---|---|---|
| E1 | Quantise the QKV input once instead of three times | −0.03 to −0.05 s | Byte-identical Q8_1 input |
| E2 | Fuse residual add + LayerNorm and emit Q8_1 directly from the norm | −0.05 to −0.10 s | Must keep the per-row reduction order |
| E3 | FC1 epilogue: tanh-GELU + Q8_1 quantise for FC2 in the MMQ output stage | −0.05 to −0.08 s | Removes the GELU and FC2-quantise passes |
| E4 | Attention beyond q48: K/V tile 64, register-blocked score/output, subgroup-free | 0–10% of 1.08 s | q64 already regressed (register pressure); needs the attribution harness |
| E5 | Fused QKV MMQ (N = 3840) | unknown | Rejected on the F32 path (+7..14%); untested on MMQ |
| E6 | Conv front end as im2col + MMQ | −0.05 s | 0.115 s fenced |
| E7 | Dispatch-gap audit (~460 dispatches per window) | unknown | Measure GPU idle with timestamps first |

Realistic combined gain: E1–E3 and E6 together are about −0.2 s per window (−7–8%).

## 3. Decoder (CPU, ~7 ms per token)

Per token: LM head ~2 ms, cross-attention ~2 ms, MLP ~2 ms, self-attention ~1 ms. Q5 kernels are compute-bound: 4-thread throughput is 22–30 Gelem/s, against a 45 GB/s stream ceiling.

| # | Item | Estimate | Notes |
|---|---|---|---|
| D1 | Pipeline encoder(n+1) ‖ decoder(n) when windows are known up front (VAD groups or fixed seeks) | −0.2 to −1.0 s per window on multi-window audio | Exact: the encoder does not depend on decoded text; only the prompt does |
| D2 | Further Q5 unpack work (VPSHUFB bit expansion, 2-row interleave) | −5–10% of decoder | Same per-row order |
| D3 | AVX2 `fastExpF32` softmax (separate mul/add, scalar sequential sum) | ~−0.1 ms per token | Exact on GOAMD64=v1 builds |
| D4 | Cross-attention on GPU (K/V already resident) | up to −1.5 ms per token | Must replicate Sdotx4/fastExp/Saxpy bits; per-token dispatch latency risk |
| D5 | F16 cross K/V (as whisper.cpp stores them) | about −1 ms per token | **Requalify**: changes bits, though it moves closer to the original |

## 4. Quality and behaviour (needed before app adoption)

- Groups VAD segmentation: Go produces 4 segments where the original produces 2, with the same words.
- Compact VAD with word timing fails closed on words that cross removed gaps; it needs a defined policy, such as gap-preserving windows for word jobs.
- Long-file accuracy and throughput on Vulkan Whisper (for example the 77-minute interview) are not measured yet.
- Nemotron speaker labels: 0 words labelled on JFK under the current overlap policy.

## Suggested order

1. Decide GPU access for the app service: it shares the Intel GPU with Qwen.
2. If yes: packed-only GGML load, a public fast-path constructor and app profiles (Whisper Vulkan ASR, with Nemotron or Community-1 speakers), then deploy behind new profiles first.
3. D1 pipelining and E1–E3 fusions: about 10% more on long files.
4. Long-file accuracy run, and the remaining quality items.
