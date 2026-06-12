# Measured end-to-end RTF — pure-Go int8 encoder (large-v3, 30s clip)

Real measurements this session on the MilkV Jupiter 2 (SpaceMIT K3, 8x X60),
large-v3 safetensors, 30s podcast window, via `cmd/audio/whisper -size large-v3
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

## 8-thread result + the bandwidth ceiling (decisive)

| Config | Threads | convstem | linear | attn | other | encoder | RTF (full dec) |
|--------|---------|----------|--------|------|-------|---------|------|
| int8   | 8       | 0.4s     | 15.6s  | 10.0s| 5.0s  | ~31.0s  | 3.0  |

8 threads is **stable** for the int8 path (board survived; the f32 RVV load was
what caused earlier brownouts). Scaling 6->8 is diminishing: linear barely moves
(16.2->15.6s) while attn scales (13.7->10.0s) — i.e. **linear is memory-bound,
attn is compute-bound**.

Effective linear throughput: ~944 GMAC of encoder matmuls in 15.6s = **~60
GMAC/s** — below the 100 GMAC/s microbenchmark, the gap being per-call quant/
dequant + orchestration. Both `npu/rvv` (vmacc) and `ime2` (vmadot) plateau here
because **the DRAM bandwidth wall, not compute, is the limiter**.

### The EP's real advantage = TCM-resident GEMM
The EP reaches 19.7s @ 8T by staging weight/activation tiles in **TCM** (3 MiB
on-chip SRAM) and running the RVV kernels on TCM-resident data, so the inner
loops never touch DRAM. That is the single lever that breaks the ~60 GMAC/s wall.
`npu/tcm.go` already maps TCM from pure Go; the remaining big piece is a
TCM-tiled GEMM: DMA a weight panel + activation block into TCM, run kernelM4N32
on TCM addresses, DMA results out. This — not more threads or kernel micro-opt —
is the path from 31s to ~20s encoder (=> pure-Go RTF < 1 with turbo decode).

### Smaller wins
- Share activation quant across q/k/v/out (same `normed` input, currently
  re-quantized 3x) — ~0.5s.
- Vectorize the f32 "other" ops (layernorm/GELU/residual, 5.0s).
