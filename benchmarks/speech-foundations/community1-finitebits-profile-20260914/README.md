# Community-1 CPU stopping profile after finitebits — 14 September 2026

The pushed `d2cda9b` exact float32 finiteness predicate was profiled on the canonical overlapped 60-second AMI SDM fixture. The nested result remained byte-identical at `b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798`.

The profiled pipeline completed in 68.831 s (69.01 s process wall), with 159,144 KiB maximum RSS and zero swaps. Go sampled 78.65 s of CPU:

| Node | Flat | Flat share | Cumulative | Cumulative share |
|---|---:|---:|---:|---:|
| `simd/runtime.dotRowsx8Asm` | 40.34 s | 51.29% | 40.34 s | 51.29% |
| `community1.weSpeakerBlockConv` | 16.20 s | 20.60% | 58.34 s | 74.18% |
| `WeSpeakerBasicBlock.forwardObserved` | — | — | 65.61 s | 83.42% |
| segmentation branch | — | — | approximately 9.4 s | approximately 12% |

The former widened `IsNaN`/`IsInf` predicate cost is no longer a material profile node. Remaining latency is dominated by exact-order convolution arithmetic and patch construction. The faster tiled-GEMM path changes reduction order and remains disqualified by strict trained intermediate comparisons; the rejected x4 interleave changed no throughput. Further large gains therefore require materially different convolution/kernel work rather than another simple source-order-preserving cleanup.

Profiling perturbs timing; the unprofiled 68.338 s run remains the retained benchmark. This profile records the stopping condition and makes no new qualification claim.
