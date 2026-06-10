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
readiness: text_ready=true vision_inventory=true runtime_ready=false observed_layers=30/30 layer_tensors=655/655 missing_layer_tensors=0
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


## Operation status details

`diffusiongemmainspect -json` now includes an `operation_status` array with per-operation implementation and reference-complete flags. Text output also summarizes this as `op_status: implemented=13/13 reference_complete=0/13`, making it clear that all planned text ops have scaffolds while parity remains incomplete.
## Latest scaffold CI validation

The current scaffold was validated with:

```bash
GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-scaffold \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/final_reference_dry_run.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/final_mock_run.json \
  DIFFUSIONGEMMA_MOCK_REF_OUT=/workspace/tmp/diffusiongemma/final_mock_ref.json \
  DIFFUSIONGEMMA_STATUS_OUT=/workspace/tmp/diffusiongemma/final_status.json
```

Result: passed. The reference dry-run reports `tokenizer.json` present, mock comparison reports `match=true`, and the status summary remains `ops=13/13`, `reference_complete=false`, `runtime_ready=false`.

## Latest extended scaffold validation

The current scaffold was validated with both scaffold CI and pattern mock comparison:

```bash
GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-scaffold \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/final2_reference_dry_run.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/final2_mock_run.json \
  DIFFUSIONGEMMA_MOCK_REF_OUT=/workspace/tmp/diffusiongemma/final2_mock_ref.json \
  DIFFUSIONGEMMA_STATUS_OUT=/workspace/tmp/diffusiongemma/final2_status.json

GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-mock-pattern \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi
```

Result: both passed. The default mock comparison and non-repeated mock pattern comparison report `match=true`. Status remains `ops=13/13`, `reference_complete=false`, `runtime_ready=false`.

## Latest run-status scaffold validation

After adding per-operation status to `diffusiongemmarun`, the combined scaffold CI was re-run:

```bash
GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-scaffold \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/post_run_status_reference_dry_run.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/post_run_status_mock_run.json \
  DIFFUSIONGEMMA_MOCK_REF_OUT=/workspace/tmp/diffusiongemma/post_run_status_mock_ref.json \
  DIFFUSIONGEMMA_STATUS_OUT=/workspace/tmp/diffusiongemma/post_run_status_status.json
```

Result: passed. Run JSON and inspect JSON both expose operation status; mock comparison remains `match=true`.

## Scaffold CI after full parity target addition

After adding the full-shard `diffusiongemma-parity` target, the no-weights scaffold CI was re-run:

```bash
GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-scaffold \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/parity_target_reference_dry_run.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/parity_target_mock_run.json \
  DIFFUSIONGEMMA_MOCK_REF_OUT=/workspace/tmp/diffusiongemma/parity_target_mock_ref.json \
  DIFFUSIONGEMMA_STATUS_OUT=/workspace/tmp/diffusiongemma/parity_target_status.json
```

Result: passed. This confirms the full parity target is additive and the metadata-only scaffold workflow remains green.


## Next command sequence

Use these commands depending on the available assets:

### Reproduce current no-weight scaffold

```bash
make diffusiongemma-bootstrap-scaffold \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma_bootstrap_smoke
```

### Check a full local checkpoint after downloading shards

```bash
make diffusiongemma-download \
  DIFFUSIONGEMMA_MODEL=models/diffusiongemma-26B-A4B-it \
  DIFFUSIONGEMMA_ACCEPT_LARGE_DOWNLOAD=yes

make diffusiongemma-check-weights \
  DIFFUSIONGEMMA_MODEL=models/diffusiongemma-26B-A4B-it
```

### Capture and compare parity once full weights and Python deps are available

```bash
make diffusiongemma-parity \
  DIFFUSIONGEMMA_MODEL=models/diffusiongemma-26B-A4B-it \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/reference.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/run.json
```

Expected current outcome: scaffold commands pass; full-weight/parity commands are gated on the 11 safetensor shards and a working Transformers/PyTorch environment.

## Latest scaffold CI with structured messages

The main scaffold CI now includes structured-message coverage by default. It was re-run with:

```bash
GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-scaffold \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/expanded_reference_dry_run.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/expanded_mock_run.json \
  DIFFUSIONGEMMA_MOCK_REF_OUT=/workspace/tmp/diffusiongemma/expanded_mock_ref.json \
  DIFFUSIONGEMMA_STATUS_OUT=/workspace/tmp/diffusiongemma/expanded_status.json
```

Result: passed. This now covers text scaffold readiness, reference dry-run, mock comparison, structured JSON messages, chat-template scaffold path, status JSON, and status summary in the default no-weight CI path.


## Checkpoint size metadata

The safetensors index reports:

```text
total_parameters=25823778864
total_size_bytes=51647562456 (~48.10 GiB)
```

`diffusiongemmainspect` surfaces these fields from `model.safetensors.index.json`, so download planning can show expected payload size before fetching the 11 shards.


`diffusiongemmainspect` and `diffusiongemma_status_summary.py` report `size_gib=48.10` for the published checkpoint payload.


## Shard download percentage

Shard readiness now includes `present_percent`, and the inspector/status summary print download progress as `present=N/11 (P%)`. Metadata-only snapshots report `0.0%`; a full checkpoint should report `100.0%`.


## Download planning quick check

Before starting the full ~48.10 GiB shard download, run:

```bash
make diffusiongemma-download-plan-only \
  DIFFUSIONGEMMA_MODEL=models/diffusiongemma-26B-A4B-it
```

Expected output includes:

```text
checkpoint shards=11 total_size=51647562456 bytes (48.10 GiB) parameters=25823778864
```


## Shard byte progress

Shard readiness now reports byte-level progress as `present_bytes/expected_bytes` and `present_byte_percent`, in addition to shard-count progress. Metadata-only snapshots report `bytes=0/51647562456 byte_percent=0`.


## Extra FFN/MoE norm tensor handles

The text tensor plan now includes DiffusionGemma-specific `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, and `post_feedforward_layernorm_2` handles for every text layer. This aligns tensor planning with the semantic forward bindings used by the dense-MLP/MoE branch scaffold.

## Scaffold CI after extra FFN/MoE norm handles

After adding `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_1`, and `post_feedforward_layernorm_2` to the text tensor/forward binding plans, scaffold CI was re-run:

```bash
GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-scaffold \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/ffn_norm_reference_dry_run.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/ffn_norm_mock_run.json \
  DIFFUSIONGEMMA_MOCK_REF_OUT=/workspace/tmp/diffusiongemma/ffn_norm_mock_ref.json \
  DIFFUSIONGEMMA_STATUS_OUT=/workspace/tmp/diffusiongemma/ffn_norm_status.json
```

Result: passed. Text readiness now reports `layer_tensors=655/655`.


`DIFFUSIONGEMMA_WEIGHTS_OUT` captures the `diffusiongemmainspect -open-weights -json` report during full parity workflows.


`diffusiongemma-weights-json` performs shard readiness preflight before opening weights, so `diffusiongemma-parity` uses it as the full-weight gate and artifact capture step. `diffusiongemma-check-weights` remains available as a standalone human-readable gate.


## No-weight aggregate CI target

`make diffusiongemma-ci-no-weights` runs the safe no-shard validation bundle: download-plan report, scaffold CI, and pattern mock CI. It does not download safetensor shards and is suitable for quick validation of the current scaffold state.

## Latest no-weight aggregate CI validation

The README-recommended safe aggregate target was run after documenting it:

```bash
GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-no-weights \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi \
  DIFFUSIONGEMMA_DOWNLOAD_PLAN_OUT=/workspace/tmp/diffusiongemma/readme_no_weight_plan.json \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/readme_no_weight_ref.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/readme_no_weight_run.json \
  DIFFUSIONGEMMA_MOCK_REF_OUT=/workspace/tmp/diffusiongemma/readme_no_weight_mock_ref.json \
  DIFFUSIONGEMMA_STATUS_OUT=/workspace/tmp/diffusiongemma/readme_no_weight_status.json
```

Result: passed. This validates the current no-weight workflow advertised in the README: download-plan report, scaffold CI, structured-message path, status export/summary, and both default and pattern mock comparison.

## Latest no-weight CI after shared summary update

After wiring `diffusiongemma_status_summary.py` to the shared inspector `summary` object, the safe aggregate CI was re-run:

```bash
GOTMPDIR=$PWD/.gotmp make diffusiongemma-ci-no-weights \
  DIFFUSIONGEMMA_MODEL=/workspace/tmp/diffusiongemma \
  DIFFUSIONGEMMA_PROMPT=hi \
  DIFFUSIONGEMMA_DOWNLOAD_PLAN_OUT=/workspace/tmp/diffusiongemma/shared_summary_plan.json \
  DIFFUSIONGEMMA_REF_OUT=/workspace/tmp/diffusiongemma/shared_summary_ref.json \
  DIFFUSIONGEMMA_RUN_OUT=/workspace/tmp/diffusiongemma/shared_summary_run.json \
  DIFFUSIONGEMMA_MOCK_REF_OUT=/workspace/tmp/diffusiongemma/shared_summary_mock_ref.json \
  DIFFUSIONGEMMA_STATUS_OUT=/workspace/tmp/diffusiongemma/shared_summary_status_ci.json
```

Result: passed. The summary output now includes `summary_text_scaffold`, `summary_shards`, `summary_reference_complete`, `summary_runtime_ready`, and consolidated `summary_missing` fields.


## Reference environment preflight

`make diffusiongemma-reference-env` runs `scripts/diffusiongemma_reference.py --check-env` to check Python, PyTorch, Transformers, and `DiffusionGemmaForBlockDiffusion` availability without loading weights. In the current container this reports missing `torch` and `transformers`, so full reference capture requires a separate Python environment setup.


## No-weight CI reference environment check

`make diffusiongemma-ci-no-weights` now includes `diffusiongemma-reference-env`, so the safe aggregate validation records whether the current Python environment can capture Transformers reference fixtures. Missing `torch`/`transformers` does not fail the no-weight CI; it is reported in `DIFFUSIONGEMMA_ENV_OUT`.


## CPU smoke target

After full shards are downloaded, use:

```bash
make diffusiongemma-run-cpu-smoke \
  DIFFUSIONGEMMA_MODEL=models/diffusiongemma-26B-A4B-it
```

The target uses a tiny canvas/max-new setting and is intended as the first real CPU dispatcher smoke before full parity.


## Slow CPU dispatcher guard

Full-weight `diffusiongemmarun -cpu-dispatcher` is now guarded by `-allow-slow-cpu` (or `DIFFUSIONGEMMA_ALLOW_SLOW_CPU=yes` through Makefile targets). A real-weight canvas-1 smoke hit the command timeout before completion, confirming the current correctness-first full-matrix CPU path is not practical yet. The guard prevents accidental long CPU runs while preserving an explicit opt-in for debugging.


## Caching and layer residency

`TextWeights` now has a decoded float tensor cache plus `PreloadGlobals`, `PreloadLayer`, and `PreloadLayerRange` helpers. `diffusiongemmarun -cpu-dispatcher` exposes `-preload-globals`, `-resident-layers N`, and `-eager-mmap`. Makefile CPU targets pass these through via `DIFFUSIONGEMMA_PRELOAD_GLOBALS`, `DIFFUSIONGEMMA_RESIDENT_LAYERS`, and `DIFFUSIONGEMMA_EAGER_MMAP`. Row-wise embedding/LM-head paths still stream from mmap through `RawTensorRow` to avoid full tied-embedding expansion.
