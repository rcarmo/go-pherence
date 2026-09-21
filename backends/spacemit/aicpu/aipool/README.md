# backends/spacemit/aicpu/aipool

The `aicpu` worker pool: a **TCM-aware, barrier-synchronized** pool that fans
quantized GEMM work across the K3 cores.

- `AIWorkerPool` — registers AI worker goroutines (cores 8–15 via
  `/proc/set_ai_thread`), with optional TCM B-wave activation staging.
- `AIGemmSpec` — the work-item DTO (weights/activations/scales/out).
- `GemmAIPooled` / `GemmAIPooledAdd` / `GemmAIPooledVL32` — pooled GEMM entry points.
- `Q4KPairBarrier`, `NewAIBarrier` — pair/barrier primitives.

Distinct from `ime2.WorkerPool` (a simpler generic GEMM pool); `aipool` adds the
engine-specific scheduling and TCM staging. Imports `ime2`/`rvv`/`tcm`; never
imports `aicpu` (so the engine → pool dependency stays acyclic).

## Worker ownership

`AIWorkerPool.Run` serialises submitters. `Close` waits for the admitted callback,
joins every worker, then unmaps TCM and clears its borrowed slices. Repeated Close
and Run after Close are no-ops. Callbacks must not call Run or Close on their own
pool; doing so is re-entrant deadlock. Callbacks must return normally; this is not
a panic-recovery executor. Constructor worker counts are bounded to the eight
AI-core/TCM blocks so concurrent workers cannot share an activation block.

Workers exit while still OS-thread locked: their registration/affinity changes
must not leak into Go's general thread pool. The lifecycle regression is guarded
by the existing K3 TestMain and requires `GO_PHERENCE_TEST_K3=1` plus
`/proc/set_ai_thread`; off-device cross-compilation does not validate TCM unmapping
or actual AI-core execution.
