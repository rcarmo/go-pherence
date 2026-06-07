# backends/spacemit/k3engine/aipool

The `k3engine` worker pool: a **TCM-aware, barrier-synchronized** pool that fans
quantized GEMM work across the K3 cores.

- `AIWorkerPool` — registers AI worker goroutines (cores 8–15 via
  `/proc/set_ai_thread`), with optional TCM B-wave activation staging.
- `AIGemmSpec` — the work-item DTO (weights/activations/scales/out).
- `GemmAIPooled` / `GemmAIPooledAdd` / `GemmAIPooledVL32` — pooled GEMM entry points.
- `Q4KPairBarrier`, `NewAIBarrier` — pair/barrier primitives.

Distinct from `ime2.WorkerPool` (a simpler generic GEMM pool); `aipool` adds the
engine-specific scheduling and TCM staging. Imports `ime2`/`rvv`/`tcm`; never
imports `k3engine` (so the engine → pool dependency stays acyclic).
