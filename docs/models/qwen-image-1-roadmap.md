# Qwen-Image (original) roadmap

`Qwen/Qwen-Image` is a separate original model from the implemented [Qwen Image 2.1 integration](../../model/qwenimage21/README.md). No original Qwen-Image weights have been downloaded or executed for this roadmap. `go-pherence` has no original-model tensor runtime or generation claim.

## Pinned metadata

- Public, ungated [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image/tree/75e0b4be04f60ec59a75f475837eced720f823b6), revision `75e0b4be04f60ec59a75f475837eced720f823b6`; Hub card metadata says `apache-2.0` and `text-to-image`. Review the repository's licence and any component licences before redistribution or checkpoint execution.
- Pinned `model_index.json` names `QwenImagePipeline`, `FlowMatchEulerDiscreteScheduler`, `Qwen2_5_VLForConditionalGeneration`, `Qwen2Tokenizer`, `QwenImageTransformer2DModel` and `AutoencoderKLQwenImage`. The listed repository has four text-encoder shards, nine transformer shards, and VAE weights. Only metadata/config/README files were inspected, not weight payloads.
- Transformer config specifies 60 layers, 24 heads × 128, joint attention dimension 3,584, 64 input and 16 output channels, and patch size two. The Qwen2.5-VL text encoder has 28 layers and hidden size 3,584; its vision encoder and token contract must be assessed separately for image editing. VAE config has 16 latent channels. The scheduler uses dynamic exponential FlowMatch shifting with a 0.02 terminal shift and 1,000 training timesteps.
- The model card demonstrates Diffusers text-to-image generation with 50 inference steps and `true_cfg_scale=4.0`. These are upstream example settings, not native Go quality or performance evidence. The 2.1 package's 32×32 one-step CPU proof, checkpoint pins and Qwen Research licence do **not** transfer to this Apache-2.0 original model.

## Work gates

1. Pin the exact upstream pipeline, tokenizer/processor and revisioned component manifests. Obtain explicit checkpoint/oracle and resource approval before downloading weights or running generation. Assess asset sizes, memory and model/component licences independently.
2. Capture bounded independent fixtures for text tokenisation, Qwen2.5-VL hidden-state selection, segment/position layout, timestep modulation, transformer blocks, VAE latent scaling, scheduler and CFG. Separate text-to-image from image-editing inputs and reference-image transport.
3. Admit memory and runtime for the 60-layer DiT, text encoder and VAE before implementing an owned Go scalar path. Reuse validated repository primitives only where numerics and layouts match; measure any SIMD/backend acceleration separately. Do not label a subprocess wrapper as pure-Go tensor execution.
4. Check output with calibrated numerical/visual comparisons against an independent implementation and human image-quality review. Never gate generated images on hash or byte identity. Measure end-to-end loading, peak RSS, repeated generation, cancellation and quality at representative resolution before support claims.

This is metadata inspection and implementation planning only. The original model has no native Go inference, released-model parity or production admission in this repository.
