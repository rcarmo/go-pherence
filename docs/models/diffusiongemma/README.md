# DiffusionGemma

Native text execution is implemented for Google's DiffusionGemma block-diffusion model, including a GGUF Q4_K_M path. Overall `runtime_ready` and `reference_complete` remain false: broader reference fixtures and full image-sequence vision validation are still missing. Those flags describe the complete model boundary, not the absence of a text runtime.

## Run text

The runner requires `-cpu-dispatcher` or `-gpu-dispatcher` for inference. Without either it uses a scaffold with no attached denoiser. Start by inspecting the checkpoint:

```bash
go run ./cmd/diffusiongemmainspect -model /path/to/diffusiongemma \
  -require-text-sparse-ready

go run ./cmd/diffusiongemmarun \
  -model /path/to/diffusiongemma-metadata \
  -gguf-model /path/to/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -cpu-dispatcher -allow-slow-cpu \
  -prompt 'hi' -canvas 1 -max-new 1 -denoise-steps 1 -decode
```

This is a bounded smoke command, not a quality benchmark. `-denoise-steps` changes the actual denoising loop; `-diffusion-steps` is reported separately. The published canvas limit is 256 tokens and larger requests are rejected.

## Choose the right guide

* [Runtime and backends](runtime.md) covers safetensors, GGUF, NVIDIA FP8 and K3 execution, with their flags and memory limits.
* [Vision limitations](vision.md) separates preprocessing and tensor planning from reference-validated image conditioning.
* [Validation and evidence](validation.md) explains the readiness fields and what the retained results establish.
* [Historical investigations](../../history/diffusiongemma/README.md) preserve the original implementation log, status snapshots and bounded GPU profiles.

The published Hugging Face checkpoint has 11 shards totalling 51,647,562,456 bytes (about 48.10 GiB). Its text stack has 30 layers, width 2816, vocabulary 262144, and 128 experts with 8 selected per token. Check asset and memory requirements before starting a full-weight run.

[Model support matrix](../supported-models.md) | [Documentation index](../../README.md)
