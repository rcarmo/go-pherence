# model/hunyuan3d

Hunyuan3D image-to-3D generation: a ViT image encoder/conditioner feeding a DiT
with a flow-matching sampler.

| Area | Files |
|---|---|
| Pipeline / runtime | `runtime.go`, `config.go`, `flow.go` (flow-matching sampler) |
| Image frontend | `image_preprocess.go`, `vit.go`, `conditioner.go`, `mini_dino.go` (standard mini DINO metadata binding only) |
| DiT | `dit.go` |
| Kernels | `kernels.go` (LinearFloat32, RMSNormFloat32, GELUTanh, PatchEmbed, AttentionFloat32) |
| Fixtures / inspection | `fixture_compare.go`, `stage_fixture.go`, `tensor_coverage.go`, `tensor_summary.go` |

`testdata/mini_dino_header.json` pins the standard mini DINO tensor layout (727
F16 entries). Regenerate it with `scripts/hunyuan3d_mini_dino_header.py` using
HTTP byte-range header reads. No DINO embedding parity or full checkpoint
payload is included; `RunImageToShape` still stops at the conditioner.

The `kernels.go` float32 ops are this model's local compute layer over
`backends/simd`; they are architecture-specific (patch-embed, 3D attention).
