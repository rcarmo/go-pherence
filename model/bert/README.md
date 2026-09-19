# model/bert

BERT-family encoder for text embeddings (e.g. reranking / retrieval).

| File | Role |
|---|---|
| `bert.go` | Model definition + load + forward |
| `forward_fast.go` | Optimized forward path |
| `workspace.go` | Reusable activation buffers |
| `checked.go` | Shape/validation helpers |

BERT has its own encoder pipeline. All model source lives under `model/`,
regardless of weight format; downloaded weights belong in `checkpoints/`.
