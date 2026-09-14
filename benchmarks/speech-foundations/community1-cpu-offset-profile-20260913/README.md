# Community-1 CPU profile after offset hoist — 13 September 2026

The pushed `add76c3` source-faithful convolution offset hoist was profiled on the canonical overlapped 60-second AMI SDM fixture. The nested result remained byte-identical at `b3c8456fcececc7199d79a6a846b76239dd87fd165e6a83532784ff8f0f62798`.

The profiled pipeline completed in 78.785 s (78.97 s process wall), with 160,240 KiB maximum RSS and zero swaps. Go sampled 88.71 s of CPU:

| Node | Flat | Flat share | Cumulative | Cumulative share |
|---|---:|---:|---:|---:|
| `simd/runtime.dotRowsx4Asm` | 44.84 s | 50.55% | 44.84 s | 50.55% |
| `community1.weSpeakerBlockConv` | 16.01 s | 18.05% | 64.69 s | 72.92% |
| `WeSpeakerBasicBlock.ForwardObserved` | 0.02 s | 0.02% | 75.40 s | 85.00% |

Before the hoist, the comparable profile attributed 34.16 s flat to `weSpeakerBlockConv` and 44.17 s to `dotRowsx4Asm`. The new profile therefore confirms that the retained change removed Go-side patch-indexing work rather than changing numerical kernels. The next measured target is the exact-order dot kernel/call granularity; more coordinate work is not justified.

Profiling perturbs timing. The unprofiled 78.710 s candidate run remains the retained benchmark. This profile selects a target and makes no new qualification claim.
