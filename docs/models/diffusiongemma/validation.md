# DiffusionGemma validation

`Capabilities()` in `model/diffusiongemma/capabilities.go` and the operation status table are the source for support reporting. Text semantic operations are marked implemented and reference-complete; the global report still lists broader reference parity and full image-sequence vision fixtures as missing.

| Field | Current value | Meaning |
|---|---|---|
| `text_only_scaffold_ready` | true | Text contracts and scaffolding are present |
| `text_full_stack_sparse_ready` | true | The sparse native text path exists |
| `sparse_topk_lm_head` | true | Sparse/debug LM-head execution is available |
| `reference_complete` | false | The complete model has outstanding reference coverage |
| `runtime_ready` | false | The overall multimodal readiness gate is not met |

Run model-free checks without loading the large checkpoint:

```bash
go test ./model/diffusiongemma -run 'TestCapabilities|TestCPUDispatcherValidate|TestEmbedCanvas|TestAppendEncoderKV' -count=1
go run ./cmd/diffusiongemmarun -h
go run ./cmd/diffusiongemmainspect -h
```

A package check can still fail on a host assembly/vet defect; do not disable vet and then describe the result as a complete validation pass. See [repository validation](../../validation/validation-gates.md) for known failures and platform checks.

## Retained results

The [implementation log](../../history/diffusiongemma/implementation-log.md) and [status snapshot](../../history/diffusiongemma/diffusiongemma-status.md) retain full-weight safetensor sparse runs through the 256-token canvas with one and two denoising steps. They record the implementation at the time of each run. Later encoder/decoder separation, masking and self-conditioning changes mean old token outputs cannot serve as fresh parity proof for the current tree.

The [GGUF GPU profile](../../history/diffusiongemma/diffusiongemma-gguf-gpu-profile.md) is a bounded workload measurement. K3/ARM64 cross-builds establish compilation only; native IME, RVV and GPU execution must be measured on their actual hardware. No new full-model timing or image-sequence parity run was performed for the documentation reorganisation.

[Support guide](README.md) | [Runtime](runtime.md) | [Vision](vision.md)
