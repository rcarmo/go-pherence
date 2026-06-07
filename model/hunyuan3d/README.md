# model/hunyuan3d

Hunyuan3D image-to-3D generation: a ViT image encoder/conditioner feeding a DiT
with a flow-matching sampler.

| Area | Files |
|---|---|
| Pipeline / runtime | `runtime.go`, `config.go`, `flow.go` (flow-matching sampler) |
| Image frontend | `image_preprocess.go`, `vit.go`, `conditioner.go` |
| DiT | `dit.go` |
| Kernels | `kernels.go` (LinearFloat32, RMSNormFloat32, GELUTanh, PatchEmbed, AttentionFloat32) |
| Fixtures / inspection | `fixture_compare.go`, `stage_fixture.go`, `tensor_coverage.go`, `tensor_summary.go` |

The `kernels.go` float32 ops are this model's local compute layer over
`backends/simd`; they are architecture-specific (patch-embed, 3D attention).
