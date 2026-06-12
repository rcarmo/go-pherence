# TCM-resident GEMM is a dead end for pure-Go RVV

Built and benchmarked the TCM-resident GEMM the roadmap pointed to. **It does not
work** for a CPU/RVV compute path.

## Measurements (core 0, /dev/tcm mmap)

| Access pattern              | Bandwidth   |
|-----------------------------|-------------|
| DRAM (malloc) byte reads    | ~1098 MB/s  |
| TCM byte reads (CPU load)   | **18.7 MB/s** |
| TCM byte reads + acquire + core-pinned | 18.7 MB/s (no change) |
| TCM 64-bit reads            | 148.9 MB/s  |
| TCM<->DRAM memcpy (burst)   | ~1.2 GB/s   |
| Full int8 GEMM reading B from TCM (RVV) | **>90s vs 190ms DRAM (~500x slower)** |

## Why

TCM is mapped **uncached device memory**. `O_SYNC` vs not, `TCM_ACQUIRE` vs not,
and core affinity make **no difference** to CPU/RVV read latency. Uncached loads
are catastrophic for the GEMM inner loop (each `vle8.v` is an uncached transaction).
TCM is fast only for **bulk DMA/memcpy** (~1.2 GB/s), i.e. it is a DMA staging
buffer for a hardware accelerator — not a CPU scratchpad.

Crucially, the CPU's own **L1/L2 cache already provides** the fast reuse a
scratchpad would, automatically, for DRAM-resident data. Staging into TCM
*bypasses* that cache and is strictly worse for CPU/RVV compute.

## Loop-order (cache-blocking) also doesn't move it
N-tile-outer (weight panel stays hot in L2) vs M-block-outer: ~24ms either way at
8T. At 8 threads the shared DRAM bandwidth is the hard wall (~100 GMAC/s); the
kernel is already cache-efficient enough that partition order is in the noise.

## Conclusion
- Pure-Go RVV int8 GEMM plateaus at ~100 GMAC/s -> encoder ~31s @8T -> RTF ~1.3-1.4
  with the turbo decoder. There is **no pure-Go software lever** (TCM or loop
  order) to break the DRAM-bandwidth wall.
- The EP's 19.7s must come from a hardware path that reads TCM at full speed
  (DMA + accelerator), which is **not reachable from CPU/RVV in pure Go**.
- Therefore the **EP-encoder + turbo-decoder hybrid (RTF 0.90)** remains the only
  way to beat RTF 1.0 on this board; the pure-Go RVV encoder is a ~RTF 1.3 floor.

## W4A8 (4-bit weights) also doesn't help — RVV has no native int4 MAC

Built a correct W4A8 outer-product kernel (`kernelM4N32W4`, int4 weights packed
2/byte, unpacked via vand/vxor/vadd/vsrl + 2x vsext.vf4 per k). Benchmarked vs the
int8 kernel at 8T, same [1500,1280,1280]:

| Kernel | 8T time |
|--------|---------|
| W8A8 (int8 weights) | 24.3 ms |
| W4A8 (int4 weights) | **39.3 ms (slower)** |

The half-size weight reads do not pay off: the per-k nibble unpack + widen costs
more than the saved bandwidth. This is fundamental — **RVV has no native int4
dot-product/MAC**, so 4-bit weights must be widened to int32 before `vmacc`.
Either you unpack per-use (compute cost > bandwidth saving, as measured) or
unpack-once-to-int8 (which loses the bandwidth benefit). Only hardware with a
native int4 MAC (GPU/NPU) wins here.

## Final: no pure-Go software lever reaches RTF < 1 on this board
- int8 RVV encoder: ~31s @8T (RTF ~1.3 with turbo decode) — the floor.
- TCM scratchpad: dead (uncached for CPU/RVV).
- Cache-blocking: neutral.
- W4A8: slower (no native int4 MAC).
The EP-encoder + turbo-decoder hybrid (RTF 0.90) remains the only sub-1.0 path.
