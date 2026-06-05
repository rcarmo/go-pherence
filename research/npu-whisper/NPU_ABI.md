# SpaceMIT K3 NPU — userspace ABI map (for a native-Go encoder)

Reverse-engineered on MilkV Jupiter 2 (K3, 8× X60) via strace + LD_PRELOAD on the
SpaceMIT ONNX Runtime EP (`libspacemit_ep.so.2.0.2+rc5`) running the Whisper
large-v3-turbo int8 encoder. Goal: drive the NPU int8 matrix engine directly from
Go (no ORT, no cgo).

## Device nodes

| Path | fd role | Notes |
|---|---|---|
| `/dev/tcm` | `O_RDWR\|O_SYNC` | On-chip Tightly-Coupled Memory (SRAM), mmap'd as compute scratchpad |
| `/dev/ai_dma` | `O_RDWR` | NPU DMA engine (MMIO doorbell/regs) |
| `/dev/aidma_list` | `O_RDWR` | DMA **descriptor ring** (mmap'd, 4 KB) |
| `/dev/dma_msi` | `O_RDWR` | MSI completion page (mmap'd, 4 KB) + arm ioctl |
| `/dev/shm/tcm_sync_standalone` | `O_RDWR\|O_CREAT` | **Cross-process TCM lock** (4 KB mmap) — see leak below |
| `/proc/set_ai_thread` | `O_WRONLY` | Pins the calling thread as the NPU worker (write core/thread) |

## TCM layout

`mmap(/dev/tcm)`: one **3 MB** region at off 0, then **8 per-core windows of 384 KB**
(`0x60000`) at offsets `0x0, 0x60000, 0xc0000, … 0x2a0000`.

```
TCM total = 3 MB = 8 cores × 384 KB/core
```

The **384 KB/core** budget is the hard partition limit: matmul tiles must fit per core.
This is why the attention-fused "clean encoder" subgraph failed `tcm buffer acquire`
(its tile exceeded 384 KB) while the attention-on-CPU `enc_s8` partition fit.

## ioctl ABI (magic `'c'` = 0x63)

| Call | _IOC | constant | Meaning |
|---|---|---|---|
| `/dev/tcm` nr=9, dir=RW, sz=4 | `_IOWR('c',9,u32)` | **0xC0046309** | **TCM_ACQUIRE(core_id)** — arg in/out = core 0..7; the call that throws *"tcm buffer acquire failed for core id N"* |
| `/dev/tcm` nr=7, dir=R, sz=4 | `_IOR('c',7,u32)` | **0x80046307** | TCM query/init (arg out) |
| `/dev/dma_msi` nr=0 | `_IO(0,0)` | **0x0** | Arm/wait MSI completion (returns an event handle, e.g. 3). Called **once** per session, not per matmul |

Go constants:
```go
const (
    TCM_ACQUIRE = 0xC0046309 // _IOWR('c',9,uint32) arg=core_id
    TCM_QUERY   = 0x80046307 // _IOR('c',7,uint32)
    DMA_MSI_ARM = 0x0        // _IO(0,0)
)
```

## Execution model — MMIO, not syscalls

The hot path issues **almost no syscalls**: a full encoder run (hundreds of int8
matmuls) produced only **one** `dma_msi` ioctl. Descriptors and NPU commands are
written into mmap'd memory (`aidma_list` ring + per-core TCM), kicked via
memory-mapped doorbell registers in the `ai_dma`/`aidma_list` regions, and completion
is **polled in shared memory** (the MSI page). Great for latency; means the command
format lives entirely in mmap'd memory built by the EP's "Spine" kernels.

Sequence (per session):
1. open `/dev/tcm`,`/dev/ai_dma`,`/dev/aidma_list`,`/dev/dma_msi`; create/open the shm lock.
2. mmap the 4 KB rings + the 3 MB TCM (+ per-core 384 KB views).
3. `ioctl(tcm, TCM_ACQUIRE, &core)` for each core used.
4. `ioctl(dma_msi, DMA_MSI_ARM)` once.
5. Per matmul: stage int8 weights/activations into TCM, write DMA descriptor(s) into
   the `aidma_list` ring, ring the MMIO doorbell, poll completion flag.

## The TCM leak (and reboot-free fix)

`/dev/shm/tcm_sync_standalone` holds the cross-process "which cores are acquired"
state. A process **SIGKILL'd/aborted mid-acquire leaves stale acquired bits**, so the
next run fails `tcm buffer acquire` for *every* core — the NPU appears wedged.

**Validated fix (no reboot):**
```sh
rm -f /dev/shm/tcm_sync_standalone
```
Confirmed: kill encoder mid-run → next run fails TCM acquire → `rm` the shm → encoder
runs again (20.1s, correct H). Use `npu_reset.sh`. Also: always let the runner unwind
(the C++ runner's try/catch releases cleanly on EP errors); only hard kills leak.

## Still unknown (gates the native-Go matmul)

The **descriptor / matrix-engine command encoding** written into the `aidma_list` ring
and TCM. It's pure MMIO built in-memory by the EP, so neither strace nor LD_PRELOAD
(nor an exit-time dump — the EP munmaps first) captures it. Getting it needs either:
- **live in-execution sampling** of the mmap'd ring (a polling thread, race-prone), or
- **static disassembly** of the EP's Spine kernels (`onnxruntime::spacemit::Spine*`,
  the int8 GEMM/Concat TCM impls) to decode descriptor layout + doorbell offsets, or
- vendor documentation for the K3 NPU DMA/compute descriptor format.

Everything above (device ABI, TCM management, leak handling) is the tractable 80%;
the command encoding is the remaining, undocumented 20%.
