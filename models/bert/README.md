# models/bert

BERT-family encoder for text embeddings (e.g. reranking / retrieval).

| File | Role |
|---|---|
| `bert.go` | Model definition + load + forward |
| `forward_fast.go` | Optimized forward path |
| `workspace.go` | Reusable activation buffers |
| `checked.go` | Shape/validation helpers |

A standalone (non-GGUF) model, hence under `models/` rather than `model/`.
