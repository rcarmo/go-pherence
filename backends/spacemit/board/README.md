# backends/k3

High-level **compute-backend dispatch** for the SpaceMIT K3 SoC (MilkV
Jupiter 2): backend selection, op placement, and dispatch across the available
runtimes.

| File | Role |
|---|---|
| `backend.go` | Backend interface + runtime-priority stack |
| `select.go` | Backend selection / capability probing |
| `ops.go` | Op dispatch |
| `spacemit.go` | SpaceMIT ORT / AI-core path |
| `vulkan.go` | Vulkan (PowerVR BXM-4-64) path |
| `simd.go` | Portable SIMD fallback wiring |

> This is the **dispatch layer**, distinct from `backends/spacemit/aicpu`,
> which is the pure-Go inference *engine*. They share the "k3" name (the SoC) but
> sit at different levels: `backends/k3` chooses *where* compute runs;
> `aicpu` *is* one of the pure-Go compute paths.

## A100 AI cores and core affinity

The K3 (SpaceMIT X100/A100, MilkV Jupiter 2) has **16 harts** split into two
heterogeneous clusters:

| Cores | Cluster | VLEN | TCM read | Role |
|---|---|---|---|---|
| 0–7  | X100 (efficiency) | 256  | 1.14 GB/s | scalar / general compute |
| 8–15 | A100 (AI-CPU)     | 1024 | 5.4 GB/s  | **IME2 matrix engine** (the "2 TOPS AI") |

There is **no discrete NPU**. The "AI compute" is the IME2 integer-matrix
extension on cores 8–15 (4× wider vectors + a direct SRAM/TCM fast path than the
X100 cores). Running quantized GEMMs there is what makes the K3 fast.

### The catch: cores 8–15 are fenced off

A normal login/SSH shell is **cgroup-restricted to cores 0–7**, and a plain
`taskset` / `sched_setaffinity` to cores 8–15 is **silently refused** by the
kernel. You cannot reach the A100 cores by affinity alone.

### The handshake (kernel device `/proc/set_ai_thread`)

To place a thread on an A100 core it must, in order:

1. **Lock to its OS thread** (`runtime.LockOSThread`) so the TID is stable.
2. **Write its TID to `/proc/set_ai_thread`** — the kernel unlocks scheduling on
   cores 8–15 *for that TID*.
3. **`sched_setaffinity`** to a specific core in 8–15.

Skip step 2 and the affinity call is a silent no-op — you stay on cores 0–7.
On this board `/proc/set_ai_thread` is world-writable (`--w--w--w-`), so no root
is required.

The canonical implementation is **`backends/spacemit/aicpu/aipool`**:

```go
// Single thread: register the current goroutine onto an A100 core.
import "github.com/rcarmo/go-pherence/backends/spacemit/aicpu/aipool"

aipool.RegisterAIThread(8) // LockOSThread + write TID + pin to core 8
// ... now IME2 GEMMs on this goroutine run on the A100 cluster ...
```

`RegisterAIThread` (see `ai_thread.go`) is exactly:

```go
func RegisterAIThread(coreID int) {
    runtime.LockOSThread()
    tid := syscall.Gettid()
    if f, err := os.OpenFile("/proc/set_ai_thread", os.O_WRONLY, 0); err == nil {
        f.Write([]byte(strconv.Itoa(tid)))
        f.Close()
    }
    var set unix.CPUSet
    set.Zero(); set.Set(coreID)
    unix.SchedSetaffinity(0, &set)
}
```

For inference, use the **persistent worker pool** instead of registering ad-hoc
threads — it amortizes the handshake and replaces scheduler dispatch with an
atomic spin-barrier (workers never return to the OS scheduler between tiles):

```go
pool := aipool.NewAIWorkerPool(6) // 6 workers pinned to cores 8–13
defer pool.Close()
pool.Run(func(workerID, nWorkers int) {
    // each worker is already on an A100 core; do its slice of the GEMM
})
```

### TCM (on-chip scratchpad)

The A100 fast path is fed by **`/dev/tcm`** — 8 × 384 KB = 3 MB of on-chip SRAM
(`backends/spacemit/tcm`: `tcm.Open`, `tcm.IsAvailable`, `BlockCount`,
`Slice(blockID)`). `NewAIWorkerPool` stages activations through TCM
automatically when it is available; set `IME2_TCM_ACT=0` to disable.

### Verifying placement

Confirm a worker actually landed on the A100 cluster:

```go
import "golang.org/x/sys/unix"
cpu, _ := unix.SchedGetcpu() // expect 8..15 inside a registered worker
```

or from the shell, check the process's allowed set: `grep Cpus_allowed_list
/proc/<pid>/status` (a registered process shows cores in 8–15, not just 0–7).

### Requirements / caveats

- Needs the SpaceMIT kernel interfaces **`/proc/set_ai_thread`** and
  **`/dev/tcm`** present. If `/proc/set_ai_thread` is absent the registration is
  a silent no-op and everything runs on cores 0–7 (correct, just slower).
- Relevant env switches: `IME2_TCM_ACT` (TCM staging), `IME2_INT8_TCM_B_WAVE` /
  `IME2_TCM_B_WAVE*` (B-wave double-buffering), `IME2_ATTN_AI` / `IME2_SCALAR_AI`
  (which ops go to the AI cores).

> Open question (see the Jupiter 2 review): whether the standard `llama-server`
> path performs this handshake, or whether its decode threads silently stay on
> cores 0–7. `aipool` is the path that *guarantees* A100 placement.
