# model/lfm2

LFM2 (Liquid Foundation Model 2) inference: a hybrid conv + attention + MoE
architecture. This package is organized around explicit per-stage **contracts**
(layout/shape validation) plus the runtime that satisfies them.

| Area | Files |
|---|---|
| Config / runtime | `config.go`, `context.go`, `execution.go`, `state.go`, `runtime_*.go`, `schedule.go` |
| Conv mixer | `conv_contract.go`, `conv_projection.go`, `conv_state.go` |
| Attention | `attention_contract.go`, `attention_kv.go`, `attention_projection.go`, `rope.go` |
| FFN / MoE / routing | `ffn_layout.go`, `moe_contract.go`, `router_layout.go`, `routing.go`, `norm.go` |
| Embedding / layout | `embedding_contract.go`, `embedding_layout.go` |
| Inspection / readiness | `readiness.go`, `tensors.go`, `tensor_shapes.go`, `tensor_shape_validation.go`, `fixtures.go` |

> The readiness/tensor-shape inspection layer (`readiness.go`, `tensor_shapes.go`)
> is structurally shared with `model/qwen3tts` but specialized via package-local
> `RuntimeStatus`/`ReferenceCoverage` types whose fields differ per model.
