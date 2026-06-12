# backends/spacemit/tcm

Driver for the SpaceMIT K3 **TCM** (Tightly-Coupled Memory) — the on-chip SRAM
scratchpad (8 cores × 384 KB).

Maps and acquires per-core TCM regions via ioctl (`TCM_INFO_GET`, `TCM_ACQUIRE`).
Includes the mmap-race hardening (`SetGCPercent(-1)` + `LockOSThread` +
`MAP_FIXED` + retry) needed to bind regions reliably.

## Important measured caveat

TCM is **uncached** for the CPU/RVV: scalar reads run at ~18.7 MB/s (byte) /
149 MB/s (64-bit) versus ~1.1 GB/s for DRAM. It is therefore useful **only as a
DMA staging buffer**, not as a compute scratchpad — a GEMM reading its B operand
from TCM is ~500× slower than from DRAM. See `research/aicpu-whisper/TCM_DEAD_END.md`.
