# cmd/k3 — SpaceMIT K3 backend tools

Drivers, benchmarks, and bring-up tools for the pure-Go K3 inference stack. None
carry kernel code — they drive `backends/spacemit/{ime2,rvv,tcm,aicpu}`.

| Command | Purpose |
|---|---|
| `ime2run` | Pure-Go IME2/RVV inference (thin wrapper over `aicpu.Run`) |
| `ime2test`, `testi8i4`, `verifydot` | IME/RVV kernel exercisers & correctness checks |
| `npu-tcm` | TCM substrate validator (per-core acquire + round-trip) |
| `k3run`, `k3llama` | LLaMA/GGUF inference via the k3 backend |
| `k3bench`, `k3qbench`, `k3ggmlbench`, `k3ffnblockbench`, `k3graphfusebench`, `k3ortbench`, `k3ortlayerbench` | Backend / kernel / graph benchmarks |
| `k3graphrun`, `k3ggmlplan`, `k3plandump` | GGML decode-graph run / plan dump |
