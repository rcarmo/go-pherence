# NPU TCM snapshotter — live descriptor/data capture

Tools to observe the SpaceMIT NPU's TCM activity while the EP runs the encoder.

## Key finding: TCM is cross-process observable

The DMA control nodes (`/dev/ai_dma` 64K MMIO, `/dev/aidma_list`, `/dev/dma_msi`)
give each `open()` a **private** mapping — a separate process sees zero changes
(`snapshotter.c` captured 0 records during a full encode). The EP also uses **raw
syscalls** for its mmaps, so `LD_PRELOAD` of libc `mmap`/`ioctl` logs nothing
(`mmaplog.c`); only strace/ptrace can see those syscalls.

**TCM is different.** It is fixed physical on-chip SRAM, so a second process that
`mmap`s `/dev/tcm` (offset 0, 3 MiB reserve — no acquire needed for read) sees the
**same physical memory the EP/NPU read and write**. `snaptcm.c` captured **6.8M
byte-changes** across the 3 MiB during one encode. The hottest regions are each
core's offset-0 window (probes 0/8/16/24/32/40, ~610k changes each = the
accumulator/control area); the rest are the streamed int8 weight/activation tiles.

## Captured transition (snapfirst.c, core0 offset 0)

`core0_off0_onset_sample.txt`: before compute (t=0) the window holds int8 data
(small signed bytes `05 01 fd 00 ...`); at compute onset (t≈10.35s) it transitions
to sparse **int32 accumulator outputs** (`ff 00 00 00`, `08 00 00 00`,
`03 00 00 00`) — the matmul results landing in TCM, observed live cross-process.

## Tools
- `snapshotter.c <sec>` — poll ai_dma/aidma_list/dma_msi (proves they're private).
- `snaptcm.c <sec>` — map /dev/tcm reserve, 64 spread probes, change histogram.
- `snapfirst.c <sec> <offset>` — record first 60 distinct 256-byte states of a
  TCM window from compute onset (descriptor/output evolution).
- `mmaplog.c` — LD_PRELOAD mmap/ioctl logger (shows EP uses raw syscalls).
- `probe_regions.c` — mmappability/size probe of the DMA nodes.

## Next
Locate the command/descriptor region (changes ~256x = once per matmul, vs the
churning tile/accumulator areas) and decode M/N/K + src/dst TCM offsets. The
capture path is proven; this is structured correlation work.
