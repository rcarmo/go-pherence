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
layer_tensors=655/655
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


## Text forward binding scaffold

`model/diffusiongemma/text_forward.go` adds semantic bindings over opened text weights. `TextWeights.ForwardPlan()` maps tensor names into roles such as embeddings, final norm, self-conditioning projections, per-layer norms, Q/K/V/O projections, Q/K norms, MLP projections, routers, and expert projections. Full-attention layers are allowed to omit `v_proj`, matching the published tensor layout. The inspector reports this forward plan when `-open-weights` is used with local shard files.


## Inference engine scaffold

`model/diffusiongemma/inference.go` adds `Engine` and `GenerateTokenIDs`. The engine ties metadata, optional text weights, and an injected `Denoiser` to the block-diffusion canvas loop. It can iterate over canvases until `max_new_tokens` is reached, appending accepted canvas tokens to context for the next canvas. If no denoiser is attached it returns `DiffusionGemma denoiser is not implemented`, making the missing native forward path explicit.


## Text denoiser scaffold

`model/diffusiongemma/denoiser.go` adds `TextDenoiser`, a future tensor-backed implementation of the `Denoiser` interface. It validates shape and semantic text forward bindings, but its `Denoise` method intentionally returns `DiffusionGemma text denoiser forward is not implemented` until layer math is wired. The same file adds `ForwardBufferPlan`, which reports major scratch/logit/router/expert element counts. For the published 256-token canvas, the inspector currently reports:

```text
hidden=720896
residual=720896
logits=67108864
router=32768
experts=1441792
```


## Forward op plan

`model/diffusiongemma/ops.go` adds a semantic `ForwardOpPlan` and `ForwardDispatcher` boundary for the text denoiser. The plan currently emits a prefix canvas embedding operation, 9 high-level operations per text layer, plus tail operations (`self_condition`, `final_norm`, `lm_head`). For the published 30-layer checkpoint, the inspector reports:

```text
ops ready=true
prefix_ops=2
layer_ops=270
tail_ops=2
```

`TextDenoiser` now routes `Denoise` through a dispatcher. The default dispatcher returns an explicit not-implemented error until tensor-backed CPU/SIMD layer math is added.


## CPU/SIMD dispatcher scaffold

`model/diffusiongemma/cpu_dispatcher.go` adds `CPUDispatcher`, `ForwardScratch`, and per-operation dispatch hooks for every semantic text forward op. The dispatcher allocates the major scratch buffers from `ForwardBufferPlan`, walks the `ForwardOpPlan`, and currently returns explicit per-op not-implemented errors such as `DiffusionGemma CPU/SIMD op self_attention is not implemented`. This creates the concrete boundary for adding native SIMD math op-by-op without changing the block-diffusion sampler or `Denoiser` interface.


The explicit prefix ops are `canvas_embedding` followed by `self_condition`, matching the Transformers reference where self-conditioning is applied to input embeddings before decoder layers execute.


## Canvas embedding implementation scaffold

`CPUDispatcher` now implements the `canvas_embedding` prefix op enough to fetch embedding rows lazily from `TextWeights.RawTensorRow` and decode `BF16`, `F16`, or `F32` rows into the hidden scratch buffer. This is the first concrete tensor-backed operation boundary. Later layer ops still return explicit not-implemented errors until RMSNorm, attention, MLP, router, experts, final norm, and LM head math are wired.


## Input RMSNorm hook

`CPUDispatcher` now implements the `input_norm`, `post_attention_norm`, `pre_moe_norm`, and `post_moe_norm` layer op scaffolds. It loads the rank-1 RMSNorm weight through `TextWeights.RawTensor`, decodes `BF16`/`F16`/`F32` to float32, and applies `backends/simd/runtime.RMSNormTo` row-by-row over the hidden scratch buffer. Layer scalar is also implemented as a scalar payload load and hidden-state scale. Final norm is implemented through the same SIMD RMSNorm path as layer norms. Subsequent attention/MLP/router/expert/self-conditioning/LM-head ops still return explicit not-implemented errors.


## Layer scalar hook

`CPUDispatcher` now implements `layer_scalar` by loading a scalar or single-element tensor payload and scaling the hidden scratch buffer in place. This is intentionally small but exercises the same raw tensor path future ops will use.


## Final norm hook

`CPUDispatcher` now implements the `final_norm` tail op by loading the decoder final RMSNorm weight and applying `backends/simd/runtime.RMSNormTo` row-by-row over hidden scratch.


## Dense MLP hook

`CPUDispatcher` now implements the dense MLP op using checked SIMD facades: `GemvRows` for gate/up/down projections and `GELUTanhMulTo` for the activation product. This is correctness-first scaffolding and currently loads full projection matrices through the raw safetensor path when the op executes. Router/expert MoE, attention, self-conditioning, and LM head remain explicit not-implemented boundaries.


## Router hook

`CPUDispatcher` now implements the `router` op scaffold. It loads `router.proj.weight`, runs checked SIMD `GemvRows` from hidden state to expert scores, and applies `router.scale` plus `router.per_expert_scale` when present. The result is stored in router scratch and top-k expert IDs/scores are selected per canvas position. Expert execution remains a separate future step.


## Router top-k scaffold

Router scratch now includes per-position `TopKIDs` and `TopKVals` sized by `top_k_experts`. The router op selects the top-k expert scores after applying router scale and per-expert scale. This prepares the expert execution hook without yet running expert MLPs.


## Expert execution hook

`CPUDispatcher` now implements a correctness-first `experts` op scaffold for 3D expert tensors. It expects `experts.gate_up_proj` shaped as `[experts, 2*intermediate, hidden]` and `experts.down_proj` shaped as `[experts, hidden, intermediate]`, runs checked SIMD GEMV for selected top-k experts, applies GELU(tanh)×up activation, and accumulates weighted expert outputs into hidden scratch. This still needs parity against the Transformers implementation and may need router-score normalization refinements before it is considered numerically complete.


## LM head hook

`CPUDispatcher` now implements the `lm_head` tail op using the tied `embed_tokens.weight` matrix as an output projection. It runs checked SIMD `GemvRows` for each canvas position to produce full-vocabulary logits. This is correctness-first and uses row-wise tied embedding projection to avoid materializing the full tied embedding matrix as a float32 matrix. It will still be slow for the published 262K vocabulary, so practical inference may need chunking/caching or acceleration, but the memory path is now safer.


## Self-conditioning hook

`CPUDispatcher` now implements `self_condition` as a prefix op after canvas embedding. With no previous self-conditioning signal, it applies the reference scale-free post RMSNorm to input embeddings. When a self-conditioning signal is provided, it applies pre RMSNorm, gated GELU MLP (`gate_proj`, `up_proj`, `down_proj`), adds the signal to embeddings, and applies scale-free post RMSNorm.


## Self-attention scaffold

`CPUDispatcher` now implements a correctness-first `self_attention` scaffold. It loads Q/K/V/O projection matrices, applies Q/K RMSNorm and scale-free V RMSNorm, computes bidirectional canvas-only attention, and applies the O projection through checked SIMD GEMV helpers. For full-attention layers where `v_proj` is absent, V reuses K, matching the reference layout. RoPE is now applied for sliding and full-attention layer types using the published theta/partial-rotary settings. Sliding-window masking is scaffolded for sliding layers. Limitations: encoder KV/cache concatenation is not wired yet, so this is not numerically complete against Transformers.


## Capability reporting

`model/diffusiongemma/capabilities.go` exposes a runtime capability summary used by `diffusiongemmainspect`. It marks metadata, tensor inventory, sampler, semantic ops, RoPE, sliding-window masking, and the attention scaffold as present while explicitly reporting missing reference-complete pieces: encoder KV concatenation, parity fixtures, and processor integration. `runtime_ready` remains false.


## RoPE hook

The self-attention scaffold now applies partial RoPE after Q/K normalization. Sliding layers use theta `10000` over the standard head dimension; full-attention layers use theta `1000000` and partial rotary factor `0.25` over the global head dimension. This closes the RoPE scaffold gap; reference parity still depends on encoder KV concatenation and fixtures.


## Sliding-window mask scaffold

The canvas self-attention scaffold now masks positions outside the sliding window for `sliding_attention` layers before softmax. For the published 256-token canvas and 1024-token sliding window this does not truncate canvas-only attention, but the hook is in place for larger/future canvas settings and for parity with the layer type contract.


## Encoder KV concat scaffold

`ForwardContext` now carries optional per-layer `EncoderKVLayer` entries. The self-attention scaffold validates encoder KV shape and concatenates encoder K/V before current canvas K/V in the attention score/value loops, matching the reference decoder behavior where prior prompt/cache K/V is read-only and canvas K/V is appended for the current denoising pass. This is still not reference-complete until parity fixtures and memory-efficient attention are added.


## Memory-aware LM head

The tied LM-head hook now projects logits row-by-row from `embed_tokens.weight` through `RawTensorRow`, avoiding an eager full-matrix float32 expansion of the 262K×2816 tied embedding. This is still correctness-first and slow, but it removes the largest avoidable memory blow-up in the scaffold.


## Processor/tokenizer metadata

`loader/config/diffusiongemma_processor.go` and `model/diffusiongemma/processor.go` parse `tokenizer_config.json`, `processor_config.json`, and `chat_template.jinja` metadata. The inspector now reports tokenizer class, processor class, mask/image/think control tokens, and chat-template size. Against the fetched HF metadata snapshot:

```text
tokenizer=GemmaTokenizer
processor=Gemma4Processor
mask="<mask>"
image="<|image|>"
think="<|think|>"
chat_template_bytes=17466
```

This is metadata integration only; actual tokenization/chat-template execution remains a future step.


## Tokenizer metadata

`model/diffusiongemma/tokenizer.go` reads `tokenizer.json` metadata and extracts vocab size, added-token count, and IDs for control tokens discovered through tokenizer/processor config. Against the fetched HF metadata snapshot:

```text
vocab=262144
added=24
<pad>=0
<eos>=1
<bos>=2
<mask>=4
<|think|>=98
<turn|>=106
<|image>=255999
<|image|>=258880
<image|>=258882
```

This is still metadata extraction only; full tokenization/chat-template rendering is not implemented.


## Prompt ID scaffold

`model/diffusiongemma/prompt.go` adds typed special-token ID extraction and a token-ID-level prompt builder. It can add BOS, thinking, and generation-prompt framing tokens when their IDs are available. This deliberately does not implement full tokenizer BPE or chat-template rendering yet; it is the safe handoff point for processor/tokenizer integration.


## Run scaffold CLI

`cmd/diffusiongemmarun` exercises the public `Engine.GenerateTokenIDs` path with already-tokenized prompt IDs. With no denoiser attached it exits with the expected scaffold error:

```text
DiffusionGemma denoiser is not implemented
```

This CLI is useful for checking that metadata loading and the public inference entrypoint are wired without falsely claiming generation support.


## CPU dispatcher run mode

`cmd/diffusiongemmarun` now accepts `-cpu-dispatcher`. In that mode it opens local safetensor shards, builds `TextWeights`, attaches `TextDenoiser` with `CPUDispatcher`, and calls the public engine. Against the metadata-only snapshot it fails clearly on the missing shard files; with local shards present it will execute the current CPU/SIMD scaffold until the next unimplemented or non-parity operation is reached. Default mode remains safe and reports `DiffusionGemma denoiser is not implemented`.


## Shard availability preflight

`ShardAvailabilityFromModelDir` now checks the safetensors index for expected shard filenames and reports how many are present locally before `-open-weights` or `-cpu-dispatcher` attempts to open them. The metadata-only snapshot reports `present=0/11`; a fully downloaded checkpoint should report `present=11/11`.


## Download helper

`scripts/download_diffusiongemma.py` plus Makefile targets can fetch the public Hugging Face checkpoint metadata or full shards:

```bash
make diffusiongemma-download-metadata
make diffusiongemma-download
```

Defaults:

```text
DIFFUSIONGEMMA_REPO=google/diffusiongemma-26B-A4B-it
DIFFUSIONGEMMA_MODEL=models/diffusiongemma-26B-A4B-it
```

The helper uses `HUGGINGFACE_TOKEN` when present and skips existing files unless `--force` is passed directly to the script.


## Scaffold command targets

DiffusionGemma scaffold commands are available through Makefile targets:

```bash
make diffusiongemma-inspect
make diffusiongemma-inspect-json
make diffusiongemma-run-scaffold
make diffusiongemma-run-cpu
make diffusiongemma-download-metadata
make diffusiongemma-download
```

Key overrides:

```text
DIFFUSIONGEMMA_MODEL
DIFFUSIONGEMMA_PROMPT_IDS
DIFFUSIONGEMMA_MAX_NEW
DIFFUSIONGEMMA_CANVAS
DIFFUSIONGEMMA_SEED
```

`diffusiongemma-run-scaffold` intentionally reports `DiffusionGemma denoiser is not implemented` until a denoiser is attached; `diffusiongemma-run-cpu` requires local safetensor shards and uses the current CPU dispatcher scaffold.


## Chat-template metadata

Processor metadata now includes a lightweight `chat_template.jinja` preflight. The inspector reports whether the template references system/user/assistant roles, tool support, and thinking markers. This is not a renderer; it is a readiness/diagnostic layer before implementing full Gemma chat-template execution.


## Exact vocab helper

`model/diffusiongemma/vocab.go` loads `tokenizer.json` into exact token/string lookup maps. `diffusiongemmarun -tokens` can append comma-separated exact vocab entries such as `<bos>,<mask>` to the prompt ID list. This is not BPE tokenization; unknown text pieces fail explicitly.


## Prompt framing flags

`diffusiongemmarun` now wires `BuildPromptIDs` through CLI flags:

```bash
-add-bos
-think
-generation-prompt
```

For example, `-tokens <mask> -add-bos -think` produces prompt IDs `[2 98 4]` against the fetched tokenizer metadata. This is still exact-token/ID scaffolding, not full chat-template rendering.


## Exact decode helper

`diffusiongemmarun -decode` uses the exact `tokenizer.json` vocabulary map to display prompt/generated token strings for known token IDs. This is still not full detokenization; unknown IDs are rendered as `<id:N>`.


## Run status reporting

`diffusiongemmarun` now includes runtime capabilities and shard availability in both text and JSON output. This makes automation-friendly status checks possible even while the run path ends at the expected scaffold error.


## Reference-complete status reporting

`diffusiongemmainspect` now prints the remaining `missing_reference` capability items. Current scaffold status reports only `reference parity fixtures` and `vision/token processor integration` as missing from reference-complete status, while still keeping `runtime_ready=false` until those gaps are closed and verified.


## Reference fixture helper

`scripts/diffusiongemma_reference.py` can capture Hugging Face Transformers reference outputs for parity work. Use dry-run mode to inspect local metadata, including `tokenizer.json`, without importing/loading Transformers:

```bash
make diffusiongemma-reference-dry-run DIFFUSIONGEMMA_MODEL=/path/to/model
```

When a Python environment and full weights are available, run:

```bash
make diffusiongemma-reference DIFFUSIONGEMMA_MODEL=/path/to/model DIFFUSIONGEMMA_REF_PROMPT="Why is the sky blue?"
```

The output JSON includes prompt, input IDs, output IDs, decoded text, timing, dtype, and device. This is a fixture-capture tool, not a Go test.


## Self-conditioning feedback plumbing

`GenerateCanvas` now carries an optional self-conditioning signal between denoising steps via `ForwardInput.SelfConditioning` and `ForwardOutput.SelfConditioning`. This matches the reference loop shape where logits from the previous denoising step can be converted into soft embeddings for the next step. The logits→soft-embedding conversion is now scaffolded in the CPU dispatcher using a row-wise softmax-weighted tied-embedding pass. It is correctness-first and expensive for the full vocabulary, but completes the feedback plumbing shape.


## Self-conditioning soft embedding conversion

After LM-head logits are produced, `CPUDispatcher` now builds the next self-conditioning signal by applying softmax over each logits row and accumulating a weighted sum of tied embedding rows. Like the memory-aware LM head, this uses row-wise `RawTensorRow` access to avoid eager full-matrix expansion, but it remains computationally expensive for the 262K vocabulary and needs reference parity validation.


## Text prompt tokenization

`diffusiongemmarun -prompt` now uses the existing Go Hugging Face tokenizer loader on `tokenizer.json` for plain text prompts. For example, `-prompt "hello world"` tokenizes to `[23391 1902]` against the fetched DiffusionGemma tokenizer snapshot and `-decode` can render it back through the same tokenizer. Exact-token mode (`-tokens`) remains available for control-token scaffolding. Full chat-template rendering is still separate work.


## Simple chat message scaffold

`diffusiongemmarun -message role:text` can now be repeated to build a simplified chat-style prompt. Message content is tokenized with `tokenizer.json` as `role\ncontent` and framed with available begin/end turn tokens, matching the basic `<|turn>role\n...<turn|>` template shape. When `-think` is used with a system/developer turn, `<|think|>` is inserted after the role line inside that turn, matching the reference template more closely. This is not full Jinja rendering and currently does not tokenize role labels exactly as the model template does, but it provides a safer scaffold than raw prompt IDs for simple text turns.


## Thinking placement

The simple chat scaffold now places `<|think|>` inside the system/developer turn after `role\n`, rather than before the role token. This remains a simplified approximation of `chat_template.jinja`, but the thinking marker placement now follows the reference template shape.


## Mock denoiser

`model/diffusiongemma/mock.go` adds a deterministic `MockDenoiser` for exercising the block-diffusion control loop without weights. `diffusiongemmarun -mock-token N` attaches it. Example with a tiny canvas:

```bash
go run ./cmd/diffusiongemmarun -model /workspace/tmp/diffusiongemma -prompt "hi" -mock-token 4 -canvas 2 -max-new 2 -decode
```

Expected scaffold output includes `generated=[4 4]` and `<mask><mask>`. This is not model inference; it validates sampler/control-flow plumbing.


## Mock run target

`make diffusiongemma-run-mock` exercises the tokenizer + block-diffusion control-flow scaffold without weights by attaching `MockDenoiser`. Useful overrides:

```text
DIFFUSIONGEMMA_PROMPT
DIFFUSIONGEMMA_MOCK_TOKEN
DIFFUSIONGEMMA_CANVAS
DIFFUSIONGEMMA_MAX_NEW
DIFFUSIONGEMMA_SEED
```

The default mock token is `4` (`<mask>` in the published tokenizer metadata).


## Model generation prompt header

For repeated `-message role:text` inputs, `diffusiongemmarun -generation-prompt` appends a simplified model turn header (`<|turn>model`). The `-chat-template` path now inserts special/control token IDs directly instead of tokenizing strings like `<bos>` as ordinary text, while still remaining a scaffold rather than a full Jinja renderer.


## Text-only scaffold readiness

Capability reporting now distinguishes `text_only_scaffold_ready=true` from `reference_complete=false` and `runtime_ready=false`. This means the text-only control flow and operation scaffolds are wired enough to run with mock denoisers or local shards, but the implementation is not yet validated as reference-complete DiffusionGemma inference.


## Readiness gates

`diffusiongemmainspect` now supports two distinct readiness gates:

```bash
-require-text-scaffold-ready
-require-runtime-ready
```

`-require-text-scaffold-ready` passes for the current metadata/index/text scaffold when tensor inventory and text tensor plans are ready. `-require-runtime-ready` remains stricter and fails until reference-complete DiffusionGemma inference is implemented and verified.


## Scaffold check target

`make diffusiongemma-check-scaffold` runs `diffusiongemmainspect -require-text-scaffold-ready`, exercises the mock denoiser through both plain prompt and structured chat-template scaffold paths, and runs build-only validation for the DiffusionGemma packages. It validates the current text-only scaffold without requiring full weight shards or reference-complete runtime readiness.


## Special-token-safe chat scaffold

`diffusiongemmarun -chat-template` now builds prompt IDs using exact special token IDs for BOS, begin/end turn, thinking, and the model generation header. Role/content text is still tokenized with the Go tokenizer. This avoids accidentally tokenizing special-token strings as normal text pieces.


## Structured message input

`diffusiongemmarun` now accepts structured text chat input with `-messages-json` or `-messages-file`, both using a JSON array of `{"role":"user","content":"..."}` objects. This is useful for automation and avoids shell-escaping many repeated `-message role:text` flags.


## Reusable chat prompt builder

`model/diffusiongemma/chat_prompt.go` adds `BuildTemplateChatPromptIDs`, a tokenizer-agnostic prompt-ID builder that inserts special token IDs directly and delegates normal text to a caller-provided encode function. `diffusiongemmarun -chat-template` now uses this model-level API instead of owning prompt construction logic in the CLI.


## Scaffold CI target

`make diffusiongemma-ci-scaffold` combines the text scaffold check, reference-helper dry run, mock run/reference comparison, status JSON export, and status summary. It validates metadata/readiness, mock denoising control flow, structured chat-template scaffold input, package buildability, reference fixture preflight, and the comparison utility without requiring full safetensor shards.


Metadata-only downloads include `tokenizer.json`, so `diffusiongemmarun -prompt`, `-decode`, exact vocab lookup, and scaffold CI work from a fresh metadata snapshot without needing safetensor shards.


## Shard readiness gate

`diffusiongemmainspect -require-shards-ready` fails unless every shard listed by `model.safetensors.index.json` is present locally. `make diffusiongemma-check-shards` wraps this gate. It is stricter than `-require-text-scaffold-ready`, which only needs metadata/index files.


## Reference comparison helper

`scripts/diffusiongemma_compare_reference.py` compares a Transformers reference JSON fixture against a go-pherence run JSON. It reports ID counts, match status, and first mismatch. `make diffusiongemma-compare-reference` wraps it with `DIFFUSIONGEMMA_REF_OUT`, `DIFFUSIONGEMMA_RUN_OUT`, and optional `DIFFUSIONGEMMA_COMPARE_PREFIX`. This is a parity triage utility, not a Go test.


## Run JSON capture target

`make diffusiongemma-run-mock-json` writes a mock-denoiser go-pherence run JSON to `DIFFUSIONGEMMA_RUN_OUT`. This pairs with `make diffusiongemma-compare-reference` and is useful for validating the comparison utility and future parity workflows.


## Mock comparison target

`make diffusiongemma-mock-compare` creates a tiny mock reference JSON, captures a mock go-pherence run JSON, and compares them through `scripts/diffusiongemma_compare_reference.py`. This validates the run-output and comparison workflow without real weights.


`DIFFUSIONGEMMA_MOCK_REF_OUT` controls the tiny mock reference JSON used by `diffusiongemma-mock-compare`, keeping it separate from `DIFFUSIONGEMMA_REF_OUT` used for Transformers dry-run/reference captures.


## Operation status reporting

`model/diffusiongemma/op_status.go` exposes per-operation implementation status and aggregate counts. The inspector prints this as `ops=13/13`, while `reference_complete=false` remains separate from scaffold coverage.


## Operation status details

`diffusiongemmainspect -json` now includes an `operation_status` array with per-operation implementation and reference-complete flags. Text output also summarizes this as `op_status: implemented=13/13 reference_complete=0/13`, making it clear that all planned text ops have scaffolds while parity remains incomplete.


## Status JSON target

`make diffusiongemma-status-json` writes the full `diffusiongemmainspect -json` report to `DIFFUSIONGEMMA_STATUS_OUT`. This includes metadata, shard readiness, capabilities, and the per-operation `operation_status` array for artifact capture.


## Status summary helper

`scripts/diffusiongemma_status_summary.py` reads a `diffusiongemmainspect -json` artifact and prints a compact CI-friendly summary. Use `make diffusiongemma-status-json` followed by `make diffusiongemma-status-summary`.


The scaffold CI target also writes `DIFFUSIONGEMMA_STATUS_OUT` and prints the compact status summary, so CI artifacts can include both the full JSON report and human-readable readiness summary.


## Denoising override flags

`diffusiongemmarun` now exposes diffusion/sampler controls from the reference generation config:

```bash
-denoise-steps
-t-min
-t-max
-entropy-bound
-stability
-confidence
```

These override `InferenceOptions.Denoising` and work with mock runs today; they will carry through to real denoising once the tensor-backed denoiser is reference-complete.


## Denoising Make variables

Mock/run targets pass through denoising override variables:

```text
DIFFUSIONGEMMA_DENOISE_STEPS
DIFFUSIONGEMMA_T_MIN
DIFFUSIONGEMMA_T_MAX
DIFFUSIONGEMMA_ENTROPY_BOUND
DIFFUSIONGEMMA_STABILITY
DIFFUSIONGEMMA_CONFIDENCE
```

Use these to exercise non-default sampler schedules through `make diffusiongemma-run-mock` or JSON/scaffold CI targets.


## Run output denoising summary

When denoising overrides are provided, `diffusiongemmarun` now prints the effective sampler settings (`steps`, temperature range, entropy bound, stability, confidence) in text output. JSON output already carries the same values through `options.denoising`.


## Mock token pattern

`diffusiongemmarun` now accepts `-mock-tokens`, a comma-separated deterministic token pattern for `MockDenoiser`. `DIFFUSIONGEMMA_MOCK_TOKENS` passes this through Makefile mock targets. For example, `-mock-tokens 4,2` with a four-token canvas generates `[4 2 4 2]`.


## Pattern-aware mock comparison

`diffusiongemma-mock-compare` now honors `DIFFUSIONGEMMA_MOCK_TOKENS` when creating its tiny reference JSON. For example, `DIFFUSIONGEMMA_MOCK_TOKENS=4,2` compares against `[4,2]` instead of the default repeated mock token.


## Pattern mock CI target

`make diffusiongemma-ci-mock-pattern` runs the mock comparison workflow with a non-repeated token pattern (`4,2`). It is a lightweight check that the mock denoiser, JSON run capture, and comparison helper are not accidentally hard-coded to repeated tokens.


## Run missing-reference output

`diffusiongemmarun` text output now includes `missing_reference` from runtime capabilities, matching the inspector/status JSON. This makes mock/scaffold runs self-describing about why `reference_complete=false`.


## Run operation status

`diffusiongemmarun` now includes the same per-operation `operation_status` array in JSON output as `diffusiongemmainspect`, and text output summarizes it as `op_status: implemented=13/13 reference_complete=0/13`.


## Full-weight readiness target

`make diffusiongemma-check-weights` is the next gate after metadata scaffold CI. It runs `diffusiongemmainspect -require-shards-ready -open-weights`, requiring all 11 safetensor shards to be present and verifying that text weight bindings can be opened with dtype/shape metadata. This target is expected to fail on metadata-only snapshots and pass only after `make diffusiongemma-download` or an equivalent full checkpoint download.


## Full parity target

`make diffusiongemma-parity` documents the intended full-checkpoint parity chain:

```text
diffusiongemma-check-weights
diffusiongemma-reference
diffusiongemma-run-cpu
diffusiongemma-compare-reference
```

It requires all safetensor shards, a working Transformers/PyTorch environment for reference capture, and enough memory/compute for the current correctness-first CPU path. It is not part of scaffold CI.


## Bootstrap scaffold target

`make diffusiongemma-bootstrap-scaffold` downloads metadata (including `tokenizer.json`) and then runs `diffusiongemma-ci-scaffold`. It is the quickest no-weights path to reproduce the current scaffold readiness from an empty model directory.


## Structured reference messages

`scripts/diffusiongemma_reference.py` and `make diffusiongemma-reference` now accept structured chat messages through `--messages-json` / `DIFFUSIONGEMMA_REF_MESSAGES_JSON` or `--messages-file` / `DIFFUSIONGEMMA_REF_MESSAGES_FILE`. This aligns reference fixture capture with `diffusiongemmarun -messages-json` / `-messages-file`.


## Structured-message CI target

`make diffusiongemma-ci-structured-messages` runs the reference dry-run plus go-pherence scaffold paths with a shared JSON message payload (`DIFFUSIONGEMMA_MESSAGES_JSON`). This keeps structured-message handling exercised separately from plain prompt and exact-token paths.


## Download planning target

`make diffusiongemma-download-plan` downloads/refreshes metadata, runs inspection, exports status JSON, and prints the compact status summary. It does not download the 11 safetensor shards; use it before a full download to confirm model metadata and shard readiness.


## Checkpoint size metadata

The safetensors index reports:

```text
total_parameters=25823778864
total_size_bytes=51647562456 (~48.10 GiB)
```

`diffusiongemmainspect` surfaces these fields from `model.safetensors.index.json`, so download planning can show expected payload size before fetching the 11 shards.


## Status summary checkpoint size

`diffusiongemma_status_summary.py` now prints `parameters` and `size_bytes` from the safetensors index metadata, making the compact summary useful for download planning as well as readiness checks.


`diffusiongemmainspect` and `diffusiongemma_status_summary.py` report `size_gib=48.10` for the published checkpoint payload.


## Shard download percentage

Shard readiness now includes `present_percent`, and the inspector/status summary print download progress as `present=N/11 (P%)`. Metadata-only snapshots report `0.0%`; a full checkpoint should report `100.0%`.


## Status refresh target

`make diffusiongemma-status-refresh` downloads/refreshes metadata, writes `DIFFUSIONGEMMA_STATUS_OUT`, and prints the compact status summary without running mock generation. It is the quickest metadata/download readiness check.


## Download plan-only mode

`scripts/download_diffusiongemma.py --plan-only` and `make diffusiongemma-download-plan-only` download/read metadata and print the planned shard count, total bytes, GiB, and parameter count without downloading the 11 safetensor shards. Use this before `make diffusiongemma-download` to confirm the expected ~48.10 GiB payload.


## Large download guard

`make diffusiongemma-download` refuses the ~48.10 GiB safetensor shard download unless `DIFFUSIONGEMMA_ACCEPT_LARGE_DOWNLOAD=yes` is set. Use `make diffusiongemma-download-plan-only` or `make diffusiongemma-status-refresh` for metadata-only checks.


## Disk-space preflight

`download_diffusiongemma.py --plan-only` now prints target filesystem free space and `enough_space=true/false` for the published shard payload size. This helps avoid starting the guarded full download on a filesystem without enough room.


## Download plan JSON

`make diffusiongemma-download-plan-json` writes a machine-readable shard/download plan to `DIFFUSIONGEMMA_DOWNLOAD_PLAN_OUT`. The JSON includes shard count, total bytes/GiB, parameter count, target free bytes/GiB, and `enough_space`.


## Download plan summary

`make diffusiongemma-download-plan-summary` prints a compact human-readable summary from `DIFFUSIONGEMMA_DOWNLOAD_PLAN_OUT`, complementing the machine-readable `diffusiongemma-download-plan-json` target.


## Download plan report target

`make diffusiongemma-download-plan-report` writes the machine-readable download plan JSON and immediately prints the compact download-plan summary. It is the recommended preflight before opting into the full ~48.10 GiB shard download.


## Shard byte progress

Shard readiness now reports byte-level progress as `present_bytes/expected_bytes` and `present_byte_percent`, in addition to shard-count progress. Metadata-only snapshots report `bytes=0/51647562456 byte_percent=0`.


## Download status alias

`make diffusiongemma-download-status` is an alias for the metadata/status refresh path. It is intended for checking shard-count and byte-level progress during or after a partial/full checkpoint download.


## Download plan shard lists

`diffusiongemma-download-plan-json` now includes `shard_files`, `present_shards`, `missing_shards`, present/missing shard counts, `present_bytes`, and `present_byte_percent`. The compact download-plan summary prints present shard count and present byte progress as well as total size/free-space status.


## Script-level space guard

`download_diffusiongemma.py` now refuses the full shard download if target free space is below the indexed payload size, unless `--ignore-space-check` is supplied. This complements the Makefile-level `DIFFUSIONGEMMA_ACCEPT_LARGE_DOWNLOAD=yes` guard.


## Extra FFN/MoE norm tensor handles

The text tensor plan now includes DiffusionGemma-specific `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, and `post_feedforward_layernorm_2` handles for every text layer. This aligns tensor planning with the semantic forward bindings used by the dense-MLP/MoE branch scaffold.


## Weight binding JSON target

`make diffusiongemma-weights-json` first requires shard readiness, then runs `diffusiongemmainspect -open-weights -json` into `DIFFUSIONGEMMA_WEIGHTS_OUT`. It is intended for full-checkpoint environments to capture text weight dtype/shape binding metadata before parity runs.


`diffusiongemma-weights-json` performs shard readiness preflight before opening weights, so `diffusiongemma-parity` uses it as the full-weight gate and artifact capture step. `diffusiongemma-check-weights` remains available as a standalone human-readable gate.


## No-weight aggregate CI target

`make diffusiongemma-ci-no-weights` runs the safe no-shard validation bundle: download-plan report, scaffold CI, and pattern mock CI. It does not download safetensor shards and is suitable for quick validation of the current scaffold state.


## Help target

`make diffusiongemma-help` prints the safe/no-weight, full-checkpoint, and parity command groups for quick discovery.


## Shared readiness summary

`model/diffusiongemma/readiness.go` provides a shared `ReadinessSummary` used by both `diffusiongemmainspect` and `diffusiongemmarun`. It summarizes text scaffold readiness, shard readiness, reference completion, runtime readiness, and missing blockers in a single compact string and JSON object.


## Shared summary in status summary

`diffusiongemma_status_summary.py` now prints the shared `summary` object emitted by `diffusiongemmainspect -json`, including scaffold readiness, shard readiness, reference/runtime readiness, and consolidated missing blockers.


## Current implementation summary

Safe/no-weight workflow is green:

```bash
make diffusiongemma-ci-no-weights DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma
```

Current no-weight status:

```text
text_scaffold=true
ops=13/13
reference_complete=false
runtime_ready=false
shards_ready=false
```

Full real-inference workflow is intentionally gated:

```bash
make diffusiongemma-download-plan-report
make diffusiongemma-download DIFFUSIONGEMMA_ACCEPT_LARGE_DOWNLOAD=yes
make diffusiongemma-check-weights
make diffusiongemma-parity
```

Remaining blockers before declaring real DiffusionGemma inference complete:

```text
full 11-shard checkpoint availability
Transformers reference fixtures
parity against reference logits/tokens
vision/token processor integration
```


## Reference environment preflight

`make diffusiongemma-reference-env` runs `scripts/diffusiongemma_reference.py --check-env` to check Python, PyTorch, Transformers, and `DiffusionGemmaForBlockDiffusion` availability without loading weights. In the current container this reports missing `torch` and `transformers`, so full reference capture requires a separate Python environment setup.


## No-weight CI reference environment check

`make diffusiongemma-ci-no-weights` now includes `diffusiongemma-reference-env`, so the safe aggregate validation records whether the current Python environment can capture Transformers reference fixtures. Missing `torch`/`transformers` does not fail the no-weight CI; it is reported in `DIFFUSIONGEMMA_ENV_OUT`.


## CPU smoke target

`make diffusiongemma-run-cpu-smoke` is a full-weight target for after all shards are present. It runs `diffusiongemma-check-weights` and then invokes `diffusiongemmarun -cpu-dispatcher` with a tiny default prompt/canvas (`hi`, canvas 1, max-new 1). This is not part of no-weight CI and is expected to fail until the 11 shards are downloaded.


## Slow CPU dispatcher guard

Full-weight `diffusiongemmarun -cpu-dispatcher` is now guarded by `-allow-slow-cpu` (or `DIFFUSIONGEMMA_ALLOW_SLOW_CPU=yes` through Makefile targets). A real-weight canvas-1 smoke hit the command timeout before completion, confirming the current correctness-first full-matrix CPU path is not practical yet. The guard prevents accidental long CPU runs while preserving an explicit opt-in for debugging.


## Caching and layer residency

`TextWeights` now has a decoded float tensor cache plus `PreloadGlobals`, `PreloadLayer`, and `PreloadLayerRange` helpers. `diffusiongemmarun -cpu-dispatcher` exposes `-preload-globals`, `-resident-layers N`, and `-eager-mmap`. Makefile CPU targets pass these through via `DIFFUSIONGEMMA_PRELOAD_GLOBALS`, `DIFFUSIONGEMMA_RESIDENT_LAYERS`, and `DIFFUSIONGEMMA_EAGER_MMAP`. Row-wise embedding/LM-head paths still stream from mmap through `RawTensorRow` to avoid full tied-embedding expansion.


## Preload-only residency check

`diffusiongemmarun -preload-only` opens weights, applies `-eager-mmap`, `-preload-globals`, and/or `-resident-layers`, reports `float_cache_entries`, and exits before generation. `make diffusiongemma-preload-globals` uses this to validate full-shard opening plus global tensor cache residency without entering the slow full dispatcher. With the downloaded checkpoint, global preload reports `float_cache_entries=6`.


## Layer-0 residency preload

With the full checkpoint present, `diffusiongemmarun -preload-only -preload-globals -resident-layers 1` succeeds and reports `float_cache_entries=28 float_cache_bytes=6280561156 (~5.85 GiB)` (6 global tensors plus 22 layer-0 tensors). `make diffusiongemma-preload-layer0` wraps this bounded residency check.


## Decoded cache byte accounting

`TextWeights.FloatCacheBytes()` now reports decoded float32 cache memory. With full weights, `-preload-globals -resident-layers 1 -preload-only` reports `float_cache_entries=28 float_cache_bytes=6280561156` (~5.85 GiB). This confirms layer residency is expensive and must remain opt-in.


## Residency estimate

`diffusiongemmainspect -open-weights -resident-layers N` now estimates decoded float32 residency for global text tensors plus the first N layers. With full weights and `N=1`, it reports `tensors=28 float32_bytes=6280561156` (~5.85 GiB), matching `diffusiongemmarun -preload-only` cache accounting. `make diffusiongemma-residency-plan` writes the JSON report to `DIFFUSIONGEMMA_RESIDENCY_OUT`.


## Residency budget planning

`diffusiongemmainspect -open-weights -residency-budget-gib N` estimates how many decoded float32 layers can stay resident with globals under a memory budget. On the downloaded checkpoint, a 16 GiB budget fits `resident_layers=4/30` with `resident_bytes=16049700880` and about 1.05 GiB remaining. `make diffusiongemma-residency-plan` accepts `DIFFUSIONGEMMA_RESIDENCY_BUDGET_GIB`.


## Runtime layer cache eviction

`CPUDispatcher{ResidentLayerPrefix: N}` now evicts decoded tensors for completed layers outside the resident prefix as the layer loop advances. `diffusiongemmarun -resident-layers N` passes this policy to the CPU dispatcher, while `-preload-only` remains a safe way to validate the selected cache footprint without generation.


## Run-time residency budget selection

`diffusiongemmarun -residency-budget-gib N` now opens weight metadata, computes the resident layer prefix that fits under the decoded float32 budget, preloads that prefix when requested, and passes the selected `ResidentLayerPrefix` into the CPU dispatcher. On the downloaded checkpoint, `-residency-budget-gib 16 -preload-only` selects `resident_layers=4/30` and reports `float_cache_bytes=16049700880`. Make CPU targets pass this via `DIFFUSIONGEMMA_RUN_RESIDENCY_BUDGET_GIB`.


## Bounded CPU layer smoke

`diffusiongemmarun -max-dispatch-layers 1` executes prefix ops plus layer 0, skips tail/logit projection, and returns the current scaffold canvas. With full weights, `-canvas 1 -resident-layers 1 -max-dispatch-layers 1` completes and reports `float_cache_entries=22 float_cache_bytes=3256379908`; this confirms layer-0 real-weight ops execute under the decoded cache policy without attempting all 30 layers. `make diffusiongemma-run-cpu-layer1-smoke` wraps this bounded probe.


## Bounded CPU layer eviction smoke

`diffusiongemmarun -resident-layers 1 -max-dispatch-layers 2` executes layers 0 and 1, then evicts layer 1 so the retained decoded cache returns to the layer-0 footprint. With full weights it completes with `float_cache_entries=22 float_cache_bytes=3256379908`. `make diffusiongemma-run-cpu-layer2-evict-smoke` wraps this eviction probe.


## Bounded CPU budget smoke

`diffusiongemmarun -residency-budget-gib 16 -max-dispatch-layers 4` now executes the first four real-weight text layers while selecting a resident prefix from the budget planner. On the downloaded checkpoint it selects `resident_layers=4/30`, completes, and reports `float_cache_entries=88 float_cache_bytes=13025519632` (~12.13 GiB retained after execution). `make diffusiongemma-run-cpu-layer4-budget-smoke` wraps this bounded budget probe.


## Bounded CPU eviction beyond resident prefix

`diffusiongemmarun -residency-budget-gib 16 -max-dispatch-layers 8` executes the first eight real-weight text layers while retaining only the budget-selected four-layer prefix. On the downloaded checkpoint it completes with `resident_layers=4/30` and retains `float_cache_entries=88 float_cache_bytes=13025519632`, matching the four-layer footprint and confirming layers beyond the resident prefix are decoded and evicted. `make diffusiongemma-run-cpu-layer8-budget-smoke` wraps this probe.


## Bounded CPU half-stack smoke

`diffusiongemmarun -residency-budget-gib 16 -max-dispatch-layers 16` executes the first sixteen real-weight text layers while retaining only the four-layer budget-selected prefix. On the downloaded checkpoint it completes with `resident_layers=4/30` and retains `float_cache_entries=88 float_cache_bytes=13025519632`, confirming decode/evict behavior across more than half the text stack. `make diffusiongemma-run-cpu-layer16-budget-smoke` wraps this probe.


## Sparse top-k LM-head smoke

`CPUDispatcher` now supports `LMHeadTopK` and `TailAfterMaxLayers`. `diffusiongemmarun -max-dispatch-layers 1 -tail-after-max-layers -lm-head-top-k 8` executes layer 0, final norm, and a sparse top-8 tied LM-head projection that fills non-top logits with `-Inf` instead of using the dense output semantically. With the downloaded checkpoint it produces a real top-k token (`generated=[236991]`, decoded as `ை`) and reports `float_cache_entries=27 float_cache_bytes=3327771140`. This is still not reference-complete, but it is the first bounded full-weight path through tail + LM-head.


## Cached sparse LM-head projection

Sparse top-k LM-head mode now decodes `model.decoder.embed_tokens.weight` into the `TextWeights` float cache once and scans that resident matrix for top-k candidate scores. Dense LM-head mode still streams rows via `RawTensorRow`. This trades roughly 2.75 GiB of decoded embedding residency for avoiding hundreds of thousands of mmap row decode calls during top-k probes.


## SIMD sparse top-k LM-head smoke

Sparse top-k LM-head now uses the decoded cached tied embedding matrix plus `simd.GemvRows` instead of per-row mmap/decode. The 1-layer top-k smoke still returns `generated=[236991]`, and the previously timing-out four-layer top-k probe now completes with `generated=[59475]`, decoded as ` 싶`, under `-residency-budget-gib 16 -max-dispatch-layers 4 -tail-after-max-layers -lm-head-top-k 8`. `make diffusiongemma-run-cpu-layer4-topk-smoke` wraps this bounded tail+LM-head probe.


## Eight-layer sparse top-k inference smoke

`diffusiongemmarun -residency-budget-gib 16 -max-dispatch-layers 8 -tail-after-max-layers -lm-head-top-k 8` now completes with the SIMD cached LM-head path. On the downloaded checkpoint it keeps `resident_layers=4/30`, retains `float_cache_entries=94 float_cache_bytes=16049700880`, and emits `generated=[19338]` decoded as ` того`. `make diffusiongemma-run-cpu-layer8-topk-smoke` wraps this deeper bounded real-weight inference probe.


## Single-step 16-layer sparse top-k smoke

`diffusiongemmarun -denoise-steps 1 -residency-budget-gib 16 -max-dispatch-layers 16 -tail-after-max-layers -lm-head-top-k 8 -dispatch-progress` completes a single denoising step through the first sixteen real-weight text layers plus final norm and SIMD sparse top-k LM-head. On the downloaded checkpoint it emits `generated=[1595]` decoded as ` way`; progress logging shows per-layer timing and cache residency. `make diffusiongemma-run-cpu-layer16-topk-step-smoke` wraps this practical half-stack inference probe.


## Single-step full-stack sparse top-k smoke

`diffusiongemmarun -denoise-steps 1 -residency-budget-gib 16 -max-dispatch-layers 30 -tail-after-max-layers -lm-head-top-k 8 -dispatch-progress` now completes one denoising step through all thirty real-weight text layers plus final norm and SIMD sparse top-k LM-head. On the downloaded checkpoint it emits `generated=[147485]` decoded as `不开`; progress logging shows the 16 GiB budget selects a four-layer resident prefix and layer tensors outside that prefix are decoded/evicted. `make diffusiongemma-run-cpu-layer30-topk-step-smoke` wraps this full-stack single-step probe.


## Two-step full-stack sparse top-k smoke

`diffusiongemmarun -denoise-steps 2 -residency-budget-gib 16 -max-dispatch-layers 30 -tail-after-max-layers -lm-head-top-k 8 -dispatch-progress` now completes two denoising iterations through all thirty real-weight text layers plus final norm, SIMD sparse top-k LM-head, and sparse self-conditioning feedback. On the downloaded checkpoint it emits `generated=[256867]` decoded as `<unused955>`. `make diffusiongemma-run-cpu-layer30-topk-2step-smoke` wraps this first full-stack multi-step feedback probe.


## Normal full-stack sparse top-k smoke

`diffusiongemmarun -denoise-steps 1 -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress` completes the normal CPU dispatcher path with all thirty real-weight text layers, final norm, and SIMD sparse top-k LM-head without using `-max-dispatch-layers` or `-tail-after-max-layers`. On the downloaded checkpoint it emits `generated=[147485]` decoded as `不开`. `make diffusiongemma-run-cpu-full-topk-step-smoke` wraps this non-debug full-stack single-step probe.


## Normal full-stack two-step sparse top-k smoke

`diffusiongemmarun -denoise-steps 2 -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress` completes the normal CPU dispatcher path for two denoising iterations with all thirty real-weight text layers, final norm, SIMD sparse top-k LM-head, and self-conditioning feedback. On the downloaded checkpoint it emits `generated=[256867]` decoded as `<unused955>`. `make diffusiongemma-run-cpu-full-topk-2step-smoke` wraps this non-debug multi-step probe.


## Normal full-stack four-step sparse top-k smoke

`diffusiongemmarun -denoise-steps 4 -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress` completes the normal CPU dispatcher path for four denoising iterations with all thirty real-weight text layers, final norm, SIMD sparse top-k LM-head, and self-conditioning feedback. On the downloaded checkpoint it emits `generated=[220642]` decoded as ` pr�f�rable`. `make diffusiongemma-run-cpu-full-topk-4step-smoke` wraps this deeper multi-step probe.


## Normal full-stack eight-step sparse top-k smoke

`diffusiongemmarun -denoise-steps 8 -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress` completes the normal CPU dispatcher path for eight denoising iterations with all thirty real-weight text layers, final norm, SIMD sparse top-k LM-head, and self-conditioning feedback. On the downloaded checkpoint it emits `generated=[115570]` decoded as ` 제일`. `make diffusiongemma-run-cpu-full-topk-8step-smoke` wraps this deeper multi-step probe.


## Two-position canvas full-stack sparse top-k smoke

`diffusiongemmarun -canvas 2 -max-new 2 -denoise-steps 1 -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress` completes a normal CPU dispatcher pass for two canvas positions through all thirty real-weight text layers, final norm, and SIMD sparse top-k LM-head. On the downloaded checkpoint it emits `generated=[19911 73585]` decoded as `eursGrav`. `make diffusiongemma-run-cpu-full-topk-canvas2-step-smoke` wraps this multi-position probe.


## Two-position two-step full-stack sparse top-k smoke

`diffusiongemmarun -canvas 2 -max-new 2 -denoise-steps 2 -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress` completes two denoising iterations for two canvas positions through all thirty real-weight text layers, final norm, SIMD sparse top-k LM-head, and self-conditioning feedback. On the downloaded checkpoint it emits `generated=[4346 4346]` decoded as ` quite quite`. `make diffusiongemma-run-cpu-full-topk-canvas2-2step-smoke` wraps this multi-position feedback probe.


## Four-position canvas full-stack sparse top-k smoke

`diffusiongemmarun -canvas 4 -max-new 4 -denoise-steps 1 -residency-budget-gib 16 -lm-head-top-k 8 -dispatch-progress` completes a normal CPU dispatcher pass for four canvas positions through all thirty real-weight text layers, final norm, and SIMD sparse top-k LM-head. On the downloaded checkpoint it emits `generated=[6596 992 253884 165249]` decoded as ` math also윳μών`. `make diffusiongemma-run-cpu-full-topk-canvas4-step-smoke` wraps this wider-canvas probe.
