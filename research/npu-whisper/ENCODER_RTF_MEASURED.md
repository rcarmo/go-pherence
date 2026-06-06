# Measured end-to-end RTF — pure-Go int8 encoder (large-v3, 30s clip)

Real measurements this session on the MilkV Jupiter 2 (SpaceMIT K3, 8x X60),
large-v3 safetensors, 30s podcast window, via `cmd/whisper -size large-v3
-timestamps`. Decode column is the FULL 32-layer large-v3 decoder unless noted.

| Config                        | Threads | convstem | linear | attn  | other | encoder | decode | total  | RTF  |
|-------------------------------|---------|----------|--------|-------|-------|---------|--------|--------|------|
| f32 baseline                  | 4       | 1.0s     | 81.4s  | 48.2s | 4.9s  | ~135s   | 75.4s  | 230.0s | 7.67 |
| int8 (ime2 vmadot, WHISPER_INT8) | 4    | 0.5s     | 21.2s  | 18.0s | 4.8s  | ~44.5s  | 42.8s  | 106.8s | 3.56 |
| int8                          | 6       | 0.5s     | 16.2s  | 13.7s | 4.8s  | ~35.2s  | 41.4s  | 96.1s  | 3.20 |

## Key facts

- **int8 cuts encoder ~4x** (linear 81->16s, attn 48->14s). 6 threads is stable
  (board survived); the f32-load reboot risk is lower for the int8 path.
- **int8 transcript quality**: differs from f32 by exactly ONE phrase out of 530
  chars ("I'll show you" vs "I'll tell you, or I'll show you" - a speech self-
  correction). Near-equivalent, not byte-identical (inherent int8 tradeoff).
- The existing `backends/spacemit/ime2` (vmadot IME) path drives the encoder;
  it hits ~90-100 GMAC/s -- same ballpark as the new `npu/rvv` vmacc kernel,
  because both are **memory-bandwidth bound** on this board.

## Path to pure-Go RTF < 1

Target config = large-v3 int8 encoder + **turbo** decoder (4 layers, ~7s, vs the
32-layer full decoder's ~41s here):
- encoder int8 @ 6T = 35.2s + turbo decode ~7s  => ~42s, **RTF ~1.4**.
- Need encoder ~22s (EP reaches 19.7s @ 8T) to clear RTF 1.0.

Remaining encoder levers (in priority):
1. attn int8 efficiency (13.7s) — per-head Q.K^T / scores.V GEMMs.
2. linear (16.2s) — apply npu/rvv register-blocking / unify with ime2.
3. "other" 4.8s — f32 layernorm/GELU/residual vector ops (vectorize).
4. thread scaling 6->8 (EP does 8 @ 19.7s; manage board power).

The proven RTF 0.90 hybrid (EP encoder + turbo decode) remains the working
solution today; this is the roadmap to match it with a fully pure-Go encoder.
