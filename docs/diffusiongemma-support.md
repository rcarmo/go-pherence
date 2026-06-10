# DiffusionGemma support scaffold

This tracks native go-pherence support for Google's DiffusionGemma text generation model family.

## Source review

Primary announcement reviewed:

- Google blog: `DiffusionGemma: 4x faster text generation`
- Model card / weights: `google/diffusiongemma-26B-A4B-it`
- Reference implementation: Hugging Face Transformers `DiffusionGemmaForBlockDiffusion`, `DiffusionGemmaGenerationMixin`, and `DiffusionGemmaGenerationConfig`

The public Hugging Face collection currently exposes:

```text
google/diffusiongemma-26B-A4B-it
architecture: DiffusionGemmaForBlockDiffusion
model_type: diffusion_gemma
format: safetensors, 11 shards
license: apache-2.0
pipeline_tag: image-text-to-text
```

Fetched reference files for inspection under `/workspace/tmp/diffusiongemma` during implementation:

```text
config.json
generation_config.json
model.safetensors.index.json
README.md
modeling_diffusion_gemma.py
generation_diffusion_gemma.py
configuration_diffusion_gemma.py
```

## Published shape

From `config.json`:

```text
model_type=diffusion_gemma
architecture=DiffusionGemmaForBlockDiffusion
dtype=bfloat16
canvas_length=256
text hidden=2816
text layers=30
text heads=16
text kv_heads=8
text global_kv_heads=2
head_dim=256
vocab=262144
sliding_window=1024
experts=128
active experts=8
moe_intermediate=704
vision hidden=1152
vision layers=27
vision heads=16
vision soft tokens/image=280
patch_size=16
boi=255999
eoi=258882
image=258880
```

Generation defaults from `generation_config.json` / reference config:

```text
max_new_tokens=256
max_denoising_steps=48
t_min=0.4
t_max=0.8
stability_threshold=1
confidence_threshold=0.005
eos=[1, 106, 50]
pad=0
sampler=EntropyBoundSampler(entropy_bound=0.1)
```

## Reference algorithm summary

DiffusionGemma is not a conventional autoregressive decoder path. It uses:

- autoregressive prompt encoder/prefill that builds context cache;
- block-autoregressive generation over a fixed-size token canvas;
- bidirectional attention over the generation canvas;
- iterative denoising of masked/noised canvas tokens;
- entropy-bounded token acceptance;
- full renoising of tokens that are not accepted at a denoising step;
- adaptive stopping when predictions are stable and sufficiently low entropy.

The model card describes this as discrete text diffusion with multi-canvas sampling. The reference generation config names the sampler as entropy-bounded denoising, with a temperature schedule from `t_max=0.8` down to `t_min=0.4` over up to `48` denoising steps.

## Current go-pherence status

Implemented scaffold:

```text
loader/config/diffusiongemma.go
model/diffusiongemma/config.go
model/diffusiongemma/runtime.go
model/diffusiongemma/sampler.go
cmd/diffusiongemmainspect
```

The scaffold can:

- parse and validate Hugging Face `config.json`;
- parse optional `generation_config.json`;
- summarize text, MoE, vision, canvas, and token dimensions;
- explicitly report `runtime_ready=false` until native block-diffusion execution exists.

Example:

```bash
go run ./cmd/diffusiongemmainspect -model /path/to/diffusiongemma-26B-A4B-it
```

## Native implementation plan

Do not route this through llama.cpp. DiffusionGemma needs a separate runtime from the existing AR LLaMA/Gemma/Qwen decode path.

Staged implementation:

1. **Metadata/readiness** — current scaffold.
2. **Tokenizer/chat-template inspection** — load tokenizer files and identify mask/canvas/control tokens.
3. **Tensor inventory** — classify embeddings, text blocks, MoE routers/experts, bidirectional canvas decoder, cross-attention/prompt cache tensors, and vision encoder tensors.
4. **CPU reference block-diffusion loop** — implement masked canvas initialization, temperature schedule, entropy computation, token acceptance, renoising, and adaptive stopping with deterministic fixtures from Transformers.
5. **Text forward path** — adapt Gemma4/Qwen MoE pieces where dimensions match, but keep block-diffusion state separate from autoregressive KV state.
6. **Vision path** — only after text-only block diffusion is correct; handle image tokens and variable visual token budgets separately.
7. **SIMD acceleration** — route hot matmul/norm/activation/entropy paths through `backends/simd/runtime` checked APIs, with scalar fallback and architecture-specific AVX/NEON/RVV kernels where useful.

## Non-goals for the scaffold

- No generation claim yet.
- No tests added in this pass per user instruction.
- No weight download committed.
- No Python dependency in the runtime path; Python/Transformers is only a reference for fixtures/parity.


## Sampler scaffold

`model/diffusiongemma/sampler.go` implements reference-aligned non-model primitives for the future block-diffusion loop:

- linear temperature schedule (`t_min + (t_max-t_min) * cur_step/max_steps`);
- temperature application to logits;
- categorical entropy from logits;
- argmax helper;
- entropy-bound canvas acceptance (`cumulative_entropy - max_entropy <= entropy_bound` over lowest-entropy positions);
- renoising of non-accepted tokens with a caller-provided RNG;
- stable-and-confident stopping state.

These are CPU/reference scaffolding pieces only. They do not yet call a DiffusionGemma forward pass or load tensors.


## Tensor inventory scaffold

`model/diffusiongemma/tensors.go` reads `model.safetensors.index.json` without opening weight shards and classifies tensor names into coarse groups. For `google/diffusiongemma-26B-A4B-it`, the current observed index summary is:

```text
total=1047
shards=11
attention=175
decoder_embedding=1
decoder_layer=120
moe=60
norm=212
other=3
router=90
vision=386
```

This is sufficient to drive the next scaffold step: identify exact tensor naming for text decoder blocks, MoE routers/experts, prompt/canvas attention, and the vision encoder before any eager tensor loading.


## Block-diffusion runtime scaffold

`model/diffusiongemma/runtime.go` now defines the model-agnostic block-diffusion loop around a narrow `Denoiser` interface:

```go
type Denoiser interface {
    Denoise(ForwardInput) (ForwardOutput, error)
}
```

`GenerateCanvas` initializes a random canvas, runs up to `max_denoising_steps`, applies the linear temperature schedule, computes argmax and entropy per canvas position, applies entropy-bound acceptance, checks stable/confident stopping, and renoises non-accepted positions. This is the control-flow scaffold for inference; the actual DiffusionGemma forward pass and tensor loading still need to be implemented behind the `Denoiser` interface.


## Tensor readiness

`diffusiongemmainspect` now reports tensor readiness from the sharded index. For the published index snapshot, text decoder readiness is true after accounting for full-attention layers that intentionally omit `self_attn.v_proj.weight`:

```text
text_ready=true
vision_inventory=true
runtime_ready=false
observed_layers=30/30
layer_tensors=565/565
missing_layer_tensors=0
```

`runtime_ready` remains false because tensor loading and forward execution are not implemented yet.


## Metadata loader

`model/diffusiongemma/model.go` now provides `LoadMetadata(modelDir)`, a single scaffold entrypoint that combines config parsing, generation defaults, denoising config, tensor inventory, and tensor readiness. It intentionally does not load full safetensor payloads yet; future tensor-backed model construction should extend this entrypoint rather than duplicate inspector logic.


## Text tensor plan

`model/diffusiongemma/tensors.go` now builds a `TextTensorPlan` from `model.safetensors.index.json`. The plan records required global decoder handles and required per-layer handles with shard names, without loading tensor payloads. Against the published index snapshot:

```text
text_plan ready=true
globals=6
layers=30
missing=0
```

This gives the future native loader exact tensor names and shard locations for the text/block-diffusion forward path.


## Text weight binding scaffold

`model/diffusiongemma/weights.go` adds `OpenTextWeights`, a non-eager binder from `TextTensorPlan` to local sharded safetensors metadata. It opens the 11 shard files when present, binds dtype/shape metadata for planned global and per-layer text tensors, and exposes `RawTensor(name)` for future payload loading. The inspector exposes this behind `-open-weights`; normal inspection remains index-only and does not require downloading shards.
