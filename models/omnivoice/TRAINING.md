# OmniVoice training efficiency assessment

Assessed 2026-09-13. Native Go supports inference and reference encoding; training
uses the upstream Python pipeline. No native backward kernels, optimiser state,
or adapter training implementation exists here. No training job was executed
for this assessment, and no training data or configuration was changed.

## Existing upstream efficiency controls

| Control | Existing behaviour | Validation needed on the training host |
|---|---|---|
| Cached audio tokens | Extract discrete codec tokens once into WebDataset tar shards (`.npy` arrays, paired JSONL and `data.lst`) | Verify all shard members, shapes, text alignment and codec version before training |
| Length-aware batches | SDPA uses length-grouped padded batches; flex attention uses sequence packing | Inspect length distribution and dropped samples; compare padding fraction and peak VRAM |
| Token budget | `batch_tokens`, `max_batch_size`, length limits and gradient accumulation bound batches | Start with a two-step smoke run; tune for the selected GPU |
| Mixed precision | `mixed_precision: "bf16"`, optional TF32 | Check GPU support and loss stability; do not assume this CPU VM tests the GPU path |
| LoRA | PEFT rank/alpha/dropout and target modules are configurable | Check gradients, checkpoint/resume and merged-checkpoint inference |
| QLoRA | No implementation or bitsandbytes/NF4 configuration found in inspected upstream sources | Requires a separate implementation and accuracy/memory tests |

Default LoRA targets are Q/K/V/O and gate/up/down projections. Audio embeddings
and heads are fully trainable through `lora_modules_to_save`; their parameters
and optimiser state still consume memory. Adapter-only parameter counts do not
represent the complete training memory requirement.

Inference reference-code JSON (`books`, `frames`, `codes`, `transcript`, optional
`ref_rms`) is a different format from training shards. Reuse the upstream shard
extractor for training. A native cache is not a drop-in WebDataset replacement.

## Safe first training run

1. Freeze a reviewed dataset revision and separate validation examples before
   extracting features. Preserve manifests and source provenance.
2. Pin model, tokenizer and codec versions. Precompute token shards once on the
   training host and reuse them between compatible runs.
3. Inspect sample lengths, alignment, token ranges and train/dev overlap. Record
   exclusions rather than silently dropping malformed samples.
4. Start with SDPA, bounded batches, BF16 on supported hardware, and a two-step
   LoRA smoke run. Check finite loss, trainable parameter names and saved state.
5. Run the bounded pilot only after the smoke test. Measure tokens/sec, padding,
   peak VRAM, extraction time and checkpoint time separately.
6. Test resume, merge the adapter into a separate standalone checkpoint, and
   validate it through the native Go loader and speech regression suite.
7. Judge English prosody/pacing and European Portuguese accent by listening.
   ASR alone cannot approve voice quality or regional pronunciation.

The local pilot template proposes 100 steps, LoRA rank 16/alpha 32/dropout 0.05,
SDPA and BF16. It has not been executed. A single NVIDIA GPU with 24 GB VRAM,
about 32 GB host RAM and 30 GB free disk is the local planning target, not a
measured minimum. Flex attention requires a compatible GPU/PyTorch combination;
the inspected upstream guide recommends Ampere or newer. This two-vCPU N100 VM
has CPU-only Torch and no usable hardware Vulkan device, so it cannot validate
GPU training or acceleration. No hardware deployment was attempted.

## Training checkpoints and inference exports

Keep resumable trainer checkpoints, optimiser/scheduler state, RNG state and
PEFT adapters. Merge into a new standalone safetensors checkpoint for native
inference acceptance tests. Preserve the training originals.

GGUF is a prospective inference export for OmniVoice. Export metadata must
identify the architecture, tensor layouts and quantisation scheme. Smaller files
need matching quantised kernels to reduce inference cost. GGUF conversion does
not implement mixed-precision training, backward computation or QLoRA. Native
Go loading of unmerged PEFT adapters also needs a separate implementation and
parity tests; merged standalone checkpoints use the existing loader contract.

## Inspected sources

The upstream project is [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice).
The assessment used the local source snapshots and preparation records below;
it does not assert that current upstream HEAD is identical to those snapshots.

- `/workspace/tmp/omnivoice-training-research/run_finetune_lora.sh`: extraction
  stage followed by `accelerate launch`.
- `data_preparation.md`, `training.md`, `pinned-config.py`, `lora_finetuning.md`
  and `train_config_finetune_lora.json` in the same directory: shard format,
  batching, precision, LoRA settings, resume and merge contracts.
- `/workspace/projects/spock-tts/training/README.md`,
  `training/pilot-config.template.json` and `docs/TRAINING-PLAN.md`: local
  preparation state and proposed pilot. These are private workspace references;
  data and source audio are not part of this repository.
- Native code: `loader/omnivoice/prompt.go`, `weights.go`,
  `models/omnivoice/reference.go` and [implementation notes](IMPLEMENTATION.md).
