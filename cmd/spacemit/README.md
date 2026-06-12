# cmd/k3 — SpaceMIT K3 backend tools

Drivers, benchmarks, and bring-up tools for the pure-Go K3 inference stack. None
carry kernel code — they drive `backends/spacemit/{ime2,rvv,tcm,aicpu}`.

| Command | Purpose |
|---|---|
| `ime2run` | Pure-Go IME2/RVV inference (thin wrapper over `aicpu.Run`) |
| `ime2test`, `testi8i4`, `verifydot` | IME/RVV kernel exercisers & correctness checks |
| `npu-tcm` | TCM substrate validator (per-core acquire + round-trip) |
| `spacemit_run`, `spacemit_llama` | LLaMA/GGUF inference via the k3 backend |
| `spacemit_bench`, `k3qbench`, `spacemit_ggmlbench`, `spacemit_ffnblockbench`, `spacemit_graphfusebench`, `spacemit_ortbench`, `spacemit_ortlayerbench` | Backend / kernel / graph benchmarks |
| `spacemit_graphrun`, `k3ggmlplan`, `k3plandump` | GGML decode-graph run / plan dump |
