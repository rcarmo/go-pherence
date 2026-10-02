# Fewer-uop Q5_0 CPU dot kernel — 2 October 2026

The packed-Q5 CPU decoder (projections, MLP and LM head) was compute-bound. Measured on Sigma:

| Measure | 1 thread | 4 threads |
|---|---:|---:|
| `SdotQ5_0` throughput | ~7.9 Gelem/s (5.4 GB/s) | 22–26 Gelem/s (16–18 GB/s) |
| Plain stream read | 18 GB/s | 45 GB/s |

The kernel keeps `Sdot`'s exact accumulation: two 8-lane accumulators taking alternating chunks, then the same horizontal reduction. That sets a floor of about 8 cycles per block. The limit was the roughly 44 vector ALU uops spent unpacking each 32-element block.

## Change

`sdotQ5_0Asm` computes the same integer `q−16` as `nib − (~(qh>>(i−4)) & 16)`, with chunk 0 shifting `qh<<4`. That removes the shift, OR and bias uops of the direct form. It also broadcasts the f16 scale with `VPBROADCASTW` + `VCVTPH2PS` and loads each nibble half once. `VCVTDQ2PS`, the single `VMULPS` by `f16(d)`, the FMA order and the reduction are unchanged, so results are bit-identical. The block now costs about 34 uops.

- Exactness: 3,000 random rows (1–200 blocks, special f16 scales, random x scales) were bit-equal to both the old kernel and the widened `SdotQ5_0Ref`. The existing `TestSdotQ5_0MatchesWidenedSdot` passes, as do the packed-decoder and batch-prefill tests.
- Kernel speed: 3.92 → 3.08 ns per block (−21%). Single-thread throughput on decoder shapes rises from ~7.9 to 9.2–10.4 Gelem/s; with 4 workers the head gains about 14%.

## Native results (5 requests, medians; every output identical to the previous arm)

| Fixture | Before | After | Decoder before → after |
|---|---:|---:|---|
| JFK | 2.850 | **2.797** | 0.272 → 0.212 |
| PT row 0 | 2.816 | **2.806** | 0.241 → 0.234 |
| PT2 (2 windows) | 6.191 | **6.026** | 1.024 → 0.885 |
| JFK VAD+words | 3.165 | **3.107** | 0.576 → 0.471 |
| groups (gap-preserving, words) | 6.436 | **6.368** | 1.090 → 0.974 |
| podcast | 3.401 | **3.327** | 0.875 → 0.759 |
| podcast VAD+words | 4.501 | **4.303** | 1.827 → 1.584 |

Pinned whisper.cpp ran PT2 in the same window before and after the Go runs: 6.211 and 6.201 s. Go's PT2 is now about 3% faster.

Gates: `go test ./...`, race (`backends/simd/runtime`, `model/whisper`), arm64/riscv64 builds, and the `GOAMD64=v3` Q5 tests. The v3 RoPE failure is pre-existing: it reproduces without this change and comes from compiler FMA fusion.
