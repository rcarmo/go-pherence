# Native Nimble validation — 2026-09-21

This milestone ports the released Bespoke Nimble typed-scoring contract to native Go. The published model bundle and Qwen3.5-9B base are Apache-2.0. The GitHub source repository at `f136b3f75721fda4ea961f73993cc50b08488835` has no root licence file, so implementation is independently written from the model bundle's documented prompt/checkpoint contract; source files are pinned only as behavioral references.

## Pinned assets

- Nimble model revision: `594dfdcfb6f94e3d0c0db7535180d3c71689169a`.
- Adapter SHA-256: `ba7e28acb97f973e80fa51f3aa6fc6f75ea4081b89632ed45d8e5f3a1d7bfa6b` (173,188,512 bytes).
- Qwen3.5-9B base revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Prompt contract SHA-256: `a0a0f94d0f65e972bc20d088678ad3d595ff1c42b9c0d78f96034526303f63fc`.
- LoRA: rank 16, alpha 32, scale 2, no bias; 496 FP32 A/B tensors across all 32 decoder layers.

The upstream-prescribed PEFT merge produced four local BF16 shards. Go also admits the unmerged LoRA layout and validates paired A/B rank and target geometry; released runtime evidence uses the merged artifact.

## Correctness work

The first real base-checkpoint probe exposed gaps in the pre-existing diagnostic Qwen3.5 runtime. Standard Hugging Face tensors use `[out,in]` linear shapes and `[channels,1,kernel]` depthwise-convolution shapes, unlike the converted MLX artifact used by earlier smokes. The loader now admits both layouts without transposing row-major linear data.

Released parity required four additional architecture fixes:

1. Qwen3.5 decoder, attention Q/K and final norms are zero-centered RMSNorm (`1 + weight`), not ordinary RMSNorm.
2. Gated-delta recurrence decays the full K×V state, computes one `state^T*k` prediction per value dimension, and applies that shared residual as an outer product.
3. Q and K use L2 normalization followed by `1/sqrt(head_dim)` query scaling.
4. Partial RoPE frequencies use the rotary dimension (64 for the released 256-dimension head at factor 0.25) as the exponent denominator.

Short sequences of one through four tokens match Transformers' candidate decisions and remain close numerically. For the released 233-token choice prompt, merged Transformers returns `[25.625, 20.000]`; Go returns `[25.557, 19.997]`, preserving the `HIGH` answer and option probability within the declared `0.1` logit gate. The second 234-token boolean branch is covered by the same opt-in released test and returns `true`. Exact prompt token IDs (233/234), candidate IDs `[32,33]`, ordered schema output and segmented-state equivalence are independently tested.

## Runtime scope

The native scorer follows the independently validated path: one full prompt per field, projecting only candidate LM-head rows. A released two-field CPU test on the i7-12700 took 7m23s after loading and peaked at 58,852,748 KiB RSS. The layer-streamed prefill path avoids the earlier prompt-length × recurrent-state snapshot explosion, but dense model storage remains FP32. The upstream MLX shared-prefix optimization is not claimed until state-fork logits match independent execution. This is not production latency.

The tiny synthetic two-token path measures roughly 2.3–2.6 µs on the i7-12700 and 9.3 µs on native ARM64 CIX P1, with 1,128 B/op and 63 allocations/op on both. CPU profiles attribute most time and allocation to the existing Qwen3.5 full-attention layer. Package statement coverage is **90.5%** without released assets; the synthetic directory test exercises the same strict model/tokenizer/head loader boundary. Allocation optimization remains explicit follow-up work rather than a hidden acceptance claim.

The committed package tests use the prompt/logit oracle without requiring model assets. Opt-in released tests require `GO_PHERENCE_NIMBLE_MODEL` and `GO_PHERENCE_NIMBLE_TOKENIZER`. No GPU was queried or used; frozen evaluation artifacts and unrelated services were unchanged.

Nimble's published 292/324 quality result is upstream evidence, not reproduced here. The Go milestone establishes checkpoint/prompt/numerical compatibility for representative enum and boolean fields; it does not claim independent calibration or external quality validation.
