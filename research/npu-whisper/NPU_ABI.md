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

## Static analysis (Capstone) — the hardware layer is isolated and small

`/dev/tcm` is **not** owned by the 3 MB EP — it lives in a dedicated
**`libspine_tcm.so.0.2.0`** (27 KB total, **10 KB `.text`, 99.4% Capstone-decodable**,
pure scalar device code, no custom NPU opcodes). This is the real RE target for the
pure-Go hardware layer.

Exported C API (`SPINE_TCM_0`):
```
spine_tcm_runtime_is_available()
spine_tcm_runtime_layout_info()       // TCM geometry (3 MB / 8 cores × 384 KB)
spine_tcm_runtime_version() / _marker()
spine_tcm_runtime_mem_get()           // acquire (the "tcm buffer acquire" path)
spine_tcm_runtime_mem_free() / _release() / _force_release()
spine_tcm_runtime_mem_query() / _mem_info() / _mem_try_wait()
```
Strings confirm: `ioctl TCM_INFO_GET` (= our `_IOR('c',7)`), `open /dev/tcm`,
`/dev/tcm_sync_mem`, `shm_open /tcm_sync_standalone` (+ `/tmp/tcm_sync_standalone.shm`
fallback), and the mmap variants (`mmap tcm block`, `mmap sync`, `mmap shm`).

So the Go port splits cleanly:
1. **TCM/DMA substrate (tractable now):** reimplement the ~11 `spine_tcm_runtime_*`
   functions in pure Go — open `/dev/tcm`, `ioctl TCM_INFO_GET`, mmap the 3 MB region +
   per-core windows, the shm sync lock, acquire/release. 10 KB of decodable code.
2. **Matrix-engine command encoding (harder, deferred):** the int8 GEMM descriptor +
   doorbell sequence is built by the EP's `Spine*` kernels in mmap'd memory; the EP
   `.text` is 97% decodable but the compute uses some custom/RVV opcodes Capstone 5.0.7
   doesn't render. Needs continued xref RE (constants are runtime-built, not literals)
   or the K3 NPU descriptor spec.

Capstone is staged (`capstone 5.0.7`, RISCV arch, RV64+RVC) with helper scripts
`sweep.py` (skip-and-continue linear sweep), `find_ioctl.py` (immediate-constant
xref), and the live samplers (`sampler.c`).

## Pure-Go TCM substrate — DONE (npu/tcm.go)

Implemented and validated on hardware (no cgo): `npu.Open()` opens `/dev/tcm`,
`ioctl(TCM_INFO_GET)` (block size 393216), maps a contiguous 3 MiB TCM region,
acquires all **8/8 cores**, maps the `aidma_list`/`dma_msi` rings, and round-trips
data through every core's TCM window. The EP still runs cleanly afterward.

Two non-obvious requirements (cost a long debug):
1. **Per-core mmap+acquire must interleave** (`mmap(core c)` then `ioctl ACQUIRE(c)`);
   acquire binds to the most-recently-mapped offset.
2. **The driver's per-core mmap sequence must not be interleaved with *any* other
   mmap syscall.** Go's runtime (GC/heap growth) races it, causing `EACCES` on the
   2nd+ core even though the syscalls are byte-identical to a working C program.
   Fix in `Open()`: `debug.SetGCPercent(-1)` + `runtime.LockOSThread()` for the
   duration, reserve a contiguous range, `MAP_FIXED` each core into it, and retry
   any core a stray runtime mmap interrupted. With that, all 8 cores acquire.

`cmd/k3/npu-tcm` is the on-device validator.

## Next: GEMM (the remaining black box)

The int8 matrix-engine **command/descriptor encoding** is still undecoded. It is
built by the EP's `Spine*` kernels directly in mmap'd memory (TCM + `aidma_list`
ring) and kicked via MMIO doorbells — no per-op syscalls, so it is invisible to
strace/LD_PRELOAD. Recovering it needs live in-execution sampling of the ring/TCM
during a *single isolated* int8 matmul, correlated with disassembly of the EP's
GEMM dispatch (Capstone, 97% decodable). That is the next phase.

## GEMM phase — progress + remaining black box

Findings toward the int8 GEMM command encoding:

- **Isolation requires full-graph EP passes.** A single `DynamicQuantizeLinear→
  MatMulInteger` subgraph extracted verbatim from the encoder fails on the NPU
  with `kernel not found MatMulInteger` — the EP only offloads after graph-level
  weight-prepacking (`SPACEMIT_EP_ENABLE_BLOCKLAYOUT`). So a single matmul can't
  be run in isolation; capture must happen inside a full (or layer-prefix) run.
- **EP introspection knobs** (`SPACEMIT_EP_DUMP_SUBGRAPHS`, `DUMP_TENSORS`,
  `DEBUG_PROFILE`) expose the EP's *high-level* compiled subgraph
  (`gemm/spine_subgraph_sample.onnx`: 256 MatMul + 256 DynamicQuantizeLinear +
  448 DequantizeLinear + Softmax/Erf/ReduceMean — the whole encoder, input
  `[1280,1500]` transposed) and a Chrome-trace profile. They do **not** expose the
  raw NPU descriptors — those are synthesized at spine-executor runtime from this
  graph and written to mmap'd TCM/ring via MMIO.
- So decoding the descriptor format needs one of:
  1. **Live in-execution sampling** of the now-Go-mappable TCM + `aidma_list` ring
     during a full encoder run, isolating the first matmul's descriptor (the Go
     `npu` substrate gives us our own TCM mapping to sample from).
  2. **Static disassembly** of the EP's spine GEMM kernel (the function turning a
     MatMul node into descriptors + doorbell offset); Capstone is staged, 97%
     decodable, constants are runtime-built so xref-driven.

This is the deep, focused remaining work. The TCM substrate (npu/tcm.go) is the
foundation it builds on; `gemm/run_mm1.cpp` is the isolated-matmul harness (kept
for the layer-prefix capture approach).
