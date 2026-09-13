# CPU profile after AMI SDM branch overlap — 13 September 2026

The accepted overlapped CPU path was rerun on the exact 60-second AMI SDM fixture with Go CPU profiling enabled. The canonical nested result hash remained `b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798`, byte-identical to the unprofiled overlapped and sequential results.

## Profile

Profile duration was 96.79 s with 106.43 s of sampled CPU time across the two admitted branches.

| Node | Flat | Flat share | Cumulative | Cumulative share |
|---|---:|---:|---:|---:|
| `simd/runtime.dotRowsx4Asm` | 44.17 s | 41.50% | 44.17 s | 41.50% |
| `community1.weSpeakerBlockConv` | 34.16 s | 32.10% | 82.31 s | 77.34% |
| `WeSpeakerBasicBlock.ForwardObserved` | 0.02 s | 0.02% | 93.23 s | 87.60% |
| `SincNet.ForwardObserved` | 0.09 s | 0.08% | 13.45 s | 12.64% |
| `LSTM.ForwardObserved` | 0.05 s | 0.05% | 4.76 s | 4.47% |

The overlap change exposed the resident CPU WeSpeaker convolution stack as the dominant remaining cost. The previously measured host pooling/projection tail remains negligible. Future cumulative work should target convolution/dot reuse, packing and batching while preserving exact source/output parity; it should not spend complexity on the host tail.

## Run/resources

- Pipeline: 96.610 s; process wall 96.80 s.
- Maximum RSS: 165,164 KiB.
- Process swaps: 0.
- Profiled output: 51 windows, 64 training rows, 3 clusters, 17/13 turns and 77 diagnostic ties.

Profiling perturbs timing, so the unprofiled 96.755 s result remains the benchmark. `cpu.pprof` and `profile-top.txt` are retained for reproducibility. This evidence selects a target; it does not itself improve or qualify the pipeline.
