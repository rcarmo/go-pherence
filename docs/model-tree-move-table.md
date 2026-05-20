# Model tree move table

Applied by `scripts/mass_move_project_tree.py` batch 2.

| Concern | Target |
|---|---|
| Attention helpers | `model/attention` |
| RoPE wrappers | `model/rope` |
| Linear/BF16/GEMV helpers | `model/linear` |
| Mixture-of-experts | `model/moe` |
| GPU forward/KV copy hooks | `model/gpu` |
| KV staging | `model/kv` |
| LM-head chunking/policy | `model/lmhead` |
| CPU decode step | `model/decode` |
| Batch prefill | `model/prefill` |
| Layer forward path | `model/layers` |
| Shared checks | `model/checks` |
| Inference helpers | `model/inference` |
| Debug hooks | `model/debug` |
| LLaMA shared core/types | `model/core` |
| CPU hot-path benchmark | `model/bench` |
