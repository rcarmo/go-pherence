# 48-query attention tiles — 2 October 2026

The encoder's padded-extent F32 flash attention, an online-softmax kernel that never stores the score matrix, now processes 48 queries per workgroup instead of 16. Every output keeps exactly the same score, max, exponent, sum and output-FMA order, and native outputs are bit-identical.

The JFK encoder drops from 3.21 s to 2.52 s, and the JFK request from 3.506 s to **2.835 s**. Pinned whisper.cpp takes 3.018 s on JFK and 3.019 s on PT row 0. All 10 fixtures produce identical output. Defaults, services and resources are unchanged.

## Attribution

A test-only harness timed 16 identical dispatches per plan on 1500 queries × 1500 keys × 20 heads × 64 dimensions, with outputs bit-compared against the 16-query kernel. Sources, binaries and logs are under `attribution/`.

| Kernel | Queries per group | ms per layer | Output |
|---|---:|---:|---|
| padded extent (before) | 16 | 54.2 | base |
| q32 | 32 | 39.5 | bit-identical |
| **q48** | 48 | **33.1** | bit-identical |
| q64 | 64 | 54.6 | bit-identical |
| whisper.cpp flash attention (GPU timer) | — | 49.4 | — |

The 16-query kernel reused each K/V tile, and each round of tile barriers and serial row reductions, for only 16 queries. With 48 queries, each lane computes 3 queries × 2 keys of scores and 3 queries × 4 output columns, so shared-memory loads per FMA fall and the tile loads and barriers are shared three times as widely. At 64 queries the kernel slows again, probably from register pressure and lower occupancy.

## Changes

- `shaders/attention_f32_key32_padded_extent_q48.glsl`, embedded in `vulkan_attention_padded_extent.go`. The shader gate now covers 34 shaders; this one regenerates and validates.
- `VkAttentionF32` gains a query-tile field (default 16), and `NewVkAttentionKey32PaddedExtentQ48F32` dispatches ceil(seqQ/48) × heads groups. The padded-extent encoder modes use it.
- The native padded-extent test now also requires q48 to reproduce the explicit-padding oracle bits on all 10 shapes. Those cover seqQ 1–4096, seqKV 1–4095 and 1–32 heads, plus cancel/reuse and guard checks. It passed on Sigma.

## Results (5 repeats, medians; every repeat's output identical to the previous arm)

| Fixture | Before | **q48** | Encoder | whisper.cpp |
|---|---:|---:|---|---:|
| JFK | 3.506 | **2.835** | 3.213→2.521 | 3.018 |
| PT row 0 | 3.527 | **2.816** | 3.218→2.524 | 3.019 |
| FR row 0 | 3.405 | **2.711** | 3.245→2.547 | — |
| JFK VAD+words (gap-preserving, DTW words) | 3.889 | **3.167** | 3.239→2.530 | see note |
| groups (gap-preserving, 2 windows, DTW words) | 7.857 | **6.411** | 6.438→5.000 | see note |
| PT2 (2 windows) | 7.588 | **6.189** | 6.449→5.032 | 6.21 |
| podcast | 4.144 | **3.391** | 3.220→2.512 | — |
| podcast VAD+words | 5.220 | **4.519** | 3.186→2.487 | — |
| PT1 | 3.412 | **2.730** | 3.235→2.551 | — |
| silence | 0.021 | 0.021 | — | — |

**Note.** The original's VAD compacts speech into one window and reports word times from token timestamps (`TokenTimestamps=true, DTW=false`). These Go VAD arms keep gaps as separate windows and compute word timing by DTW over cross-attention, which costs an extra decoder pass. A matched compact-VAD comparison is reported separately.

## Matched compact VAD (words off)

These Go arms use default compact VAD scheduling, as the original does, with words off. The original timings are medians from the [workflow refresh](whisper-workflow-refresh-20261001.md).

| Fixture | whisper.cpp | Go | Segments |
|---|---:|---:|---|
| JFK VAD | 3.028 | **2.881** | Same text. Original [0.32–7.47], [8.19–10.37]; Go [0.322–7.462], [8.206–10.366]. |
| groups VAD | 3.257 | **3.242** | Same words. The original has 2 segments (one per group); Go splits each group into two (4 segments). |

Compact VAD with Go word timing fails closed: a DTW word that crosses a removed VAD gap has no original-timeline time. The original's `TokenTimestamps` heuristic has no such check. Word timing is therefore not compared here.
