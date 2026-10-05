# Interleaved whisper.cpp vs Go comparison and batched prompt prefill — 2 October 2026

## Interleaved comparison (commit 3a4b1f7e)

Pinned whisper.cpp (`refresh-original-final` harness: Vulkan, flash attention, reusable context, 4 threads; same libraries as the [workflow refresh](whisper-workflow-refresh-20261001.md)) and Go ran in one isolated Intel GPU window. Order was ABBA: pass A ran the original first for each fixture, pass B ran Go first. Each arm made 5 requests in one process. The table gives medians of the 10 requests per engine, including each process's first request (the original's first is about 4.4 s).

| Fixture | whisper.cpp | Go | Δ | Output |
|---|---:|---:|---:|---|
| JFK | 3.033 | **2.850** | −6.0% | identical segments, text and times |
| PT row 0 | 3.023 | **2.813** | −6.9% | identical |
| PT2 (2 windows) | 6.197 | 6.258 | +1.0% | identical |
| JFK, VAD (compact, words off) | 3.025 | **2.910** | −3.8% | same text; boundaries 7.46/8.21 vs 7.47/8.19 s |
| groups, VAD (compact, words off) | 3.276 | 3.266 | −0.3% | same words; Go 4 segments, original 2 |

Every request within an arm gave identical output. Go is faster on single-window requests. The multi-window and token-heavy cases are ties, because the Go decoder runs on the CPU at roughly 7–10 ms per token while the original's runs on the GPU at about 6.4 ms per token. Go's encoder is faster per window: 2.52 s against 2.78 s.

## Batched prompt prefill

`Decoder.AdvanceTokens` feeds prompt positions that need no logits as one batch on the packed-Q5 CPU decoder. Each Q5 weight row, and each cross-attention head's K/V, is read once for all rows. Every row runs the same functions in the same per-element order as sequential `AdvanceToken`. Configurations with F32 weights, an attention observer or an NVIDIA path keep the sequential loop. The prompt callback now takes the whole prefix.

- `TestAdvanceTokensMatchesSequential` requires the KV caches, position, last token and the next three logit vectors to be bit-identical to sequential prefill. It covers 2 and 4 heads; 1, 3 and 4 workers; default and original-compat arithmetic; and prompts of 2, 5 and 23 tokens. Two mutations both fail it: a 1-ULP change to the cross-attention scale and a causal-prefix leak.
- Natively, 10 fixtures × 5 requests gave output identical to the previous arm. PT2's second-window 62-token prompt saves about 0.06 s of decoder time (cross 0.329→0.285 s, self 0.211→0.196 s). The request median is unchanged (6.189 → 6.191 s): the prompt is only a small part of decoding. The original measured 6.226 and 6.202 s in the same window.

Gates: `go test ./...`, race (`model/whisper`), arm64/riscv64 builds and docs links.
