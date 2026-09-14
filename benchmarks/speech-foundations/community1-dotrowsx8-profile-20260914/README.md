# Community-1 CPU profile after eight-row GEMV — 14 September 2026

The pushed `7cc85fe` exact-order eight-row GEMV path was profiled on the canonical overlapped 60-second AMI SDM fixture. The nested result remained byte-identical at `b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798`.

The profiled pipeline completed in 73.881 s (74.07 s process wall), with 156,080 KiB maximum RSS and zero swaps. Go sampled 83.61 s of CPU:

| Node | Flat | Flat share | Cumulative | Cumulative share |
|---|---:|---:|---:|---:|
| `simd/runtime.dotRowsx8Asm` | 40.02 s | 47.87% | 40.02 s | 47.87% |
| `community1.weSpeakerBlockConv` | 15.80 s | 18.90% | 59.45 s | 71.10% |
| `community1.finiteBlock` | 2.43 s | 2.91% | 8.55 s | 10.23% |
| `community1.weSpeakerBlockBN` | 4.84 s | 5.79% | 7.64 s | 9.14% |
| segmentation branch | — | — | 9.39 s | 11.23% |

The eight-row kernel remains dominant. A wider kernel would exceed practical AVX2 register capacity without changing reduction structure. Redundant finiteness scans are the next measured non-kernel target, but may be removed or fused only where the existing fail-closed validation boundary is preserved.

Profiling perturbs timing; the unprofiled 73.485 s run remains the retained benchmark. This profile selects a target and makes no new qualification claim.
