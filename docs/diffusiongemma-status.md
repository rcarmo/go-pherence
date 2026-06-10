# DiffusionGemma implementation status

Generated from the current go-pherence DiffusionGemma scaffold and the fetched Hugging Face metadata snapshot under `/workspace/tmp/diffusiongemma`.

## Weight/reference target

```text
repo: google/diffusiongemma-26B-A4B-it
architecture: DiffusionGemmaForBlockDiffusion
model_type: diffusion_gemma
format: safetensors, 11 shards
reference: Hugging Face Transformers DiffusionGemmaForBlockDiffusion / DiffusionGemmaGenerationMixin
```

## Current readiness

```text
text_scaffold=true
text_ready=true
vision_inventory=true
runtime_ready=false
reference_complete=false
```

Current inspector summary:

```text
shards:    ready=false present=0/11 missing=11
readiness: text_ready=true vision_inventory=true runtime_ready=false observed_layers=30/30 layer_tensors=565/565 missing_layer_tensors=0
text_plan: ready=true globals=6 layers=30 missing=0
buffers:   hidden=720896 residual=720896 logits=67108864 router=32768 experts=1441792 top_k=8
ops:       ready=true prefix_ops=2 layer_ops=270 tail_ops=2 reason=
caps:      sampler=true ops=13/13 text_scaffold=true attention_scaffold=true rope=true sliding_mask=true encoder_kv=true reference_complete=false
missing_reference: [reference parity fixtures vision/token processor integration]
```

## Implemented/scaffolded

- HF config parsing for `diffusion_gemma`.
- Generation config parsing, including entropy-bound sampler settings.
- Processor/tokenizer metadata parsing.
- Exact tokenizer vocab lookup and decode helper.
- Basic text prompt tokenization with existing Go tokenizer.
- Simple chat prompt scaffold with special-token-safe turn framing.
- Sharded safetensors index inventory without opening shard payloads.
- Shard availability preflight.
- Text tensor plan with shard names.
- Optional non-eager text weight binder.
- Semantic text forward binding plan.
- Block-diffusion canvas loop.
- Entropy-bound acceptance, renoising, temperature schedule, stable/confident stopping.
- Self-conditioning feedback plumbing.
- CPU/SIMD dispatcher scaffold.
- Implemented CPU dispatcher hooks for:
  - canvas embedding
  - self-conditioning
  - RMSNorms
  - dense MLP
  - router score computation
  - top-k routing
  - expert MLP scaffold
  - RoPE
  - sliding-window mask scaffold
  - encoder KV concat scaffold
  - layer scalar
  - final norm
  - memory-aware tied LM head
- Mock denoiser for control-flow validation.
- Inspect/run/download/reference/compare Makefile targets.

## Mock run status

The scaffold can run the block-diffusion control flow with a deterministic mock denoiser:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/diffusiongemmarun \
  -model /workspace/tmp/diffusiongemma \
  -prompt hi \
  -mock-token 4 \
  -canvas 2 \
  -max-new 2 \
  -decode
```

Current output:

```text
prompt_ids=[2202]
prompt_tokens=[hi]
generated=[4 4]
generated_tokens=[<mask><mask>]
```

## Repeatable scaffold validation

```bash
make diffusiongemma-ci-scaffold DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma
```

This runs:

- text scaffold readiness gate;
- mock denoising control-flow run;
- structured chat-template scaffold run;
- build-only Go package validation;
- reference helper dry run;
- mock run/reference comparison.

## Remaining before real/reference-complete inference

1. Download/prepare full 11 safetensor shards locally.
2. Run `diffusiongemmainspect -open-weights` and `-require-shards-ready` against the full checkpoint.
3. Capture Transformers reference fixtures with `scripts/diffusiongemma_reference.py`.
4. Compare go-pherence run JSON against reference JSON.
5. Close numerical gaps discovered by parity, especially:
   - exact chat-template/processor behavior;
   - vision/multimodal processor path;
   - attention parity details;
   - router/expert weighting parity;
   - practical performance/memory strategy for full-vocab LM head and self-conditioning.

## Current caveat

The code is a native text-only scaffold with substantial operation coverage, but `runtime_ready=false` and `reference_complete=false` remain correct until real-weight parity fixtures pass.


## Operation status summary

The inspector now reports semantic operation coverage as `ops=13/13`, meaning every planned text-only operation has a scaffold/hook. `reference_complete=false` remains correct because those ops still need parity fixtures and full processor/vision integration.
