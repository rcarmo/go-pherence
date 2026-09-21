## Needle head training and half-width models

Needle's auxiliary heads can be trained without updating the language-model trunk. `HeadLossGrad` computes a supervised loss and gradients for one selected head; `TrainHeadStep` applies AdamW and returns a new model. The input embeddings and layer cells are computed once per call with the requested numerics, then treated as constants, matching upstream's stop-gradient boundary.

This is a head-training primitive, not a reproduction of an upstream training dataset or calibration procedure. It requires a source safetensors checkpoint containing the selected head. Deployment archives, decoded-archive checkpoints, packed execution and LoRA head training are rejected.

### Targets and objectives

The JSON `target` is always an array, even for a scalar label:

| Head | Target | Objective | Inference output |
|---|---|---|---|
| `confidence` | One finite value in `[0,1]`, such as `[1]` | Numerically stable binary cross-entropy on the logit | One raw logit |
| `router` | One integer class index, `[0]`, `[1]` or `[2]` | Cross-entropy over the three logits | Three raw logits |
| `embedding` | A finite vector matching `embedding_dim`, with squared norm within `0.001` of one | Mean squared error on the normalized embedding | Unit-normalized embedding |

`tokens` contains IDs from the checkpoint's matching tokenizer. Padding is masked as it is for head inference; an empty or all-padding sequence fails. `mask` is for language-model next-token training, not head training. Invalid targets, missing head weights, mismatched shapes and insufficient workspace return errors rather than updates.

Only the selected head's probes, gain, query, row bias and projection parameters are updated. Other heads, trunk tensors and stored calibration values are preserved. A separate optimizer is required for each parameter set. Optimizers are session-owned, not concurrent-safe, and their moments are not included in the saved checkpoint.

Training with a binary label does **not** make the confidence output a calibrated probability. Neither the API nor CLI applies router calibration thresholds. After changing weights, validation and calibration on appropriate held-out data are still required; `calibrated:false` in CLI output is intentional.

### CLI examples

Build from the repository root:

```sh
go build -o bin/needle ./cmd/needle
```

A confidence example, using illustrative token IDs rather than a quality-evaluation prompt:

```sh
printf '{"tokens":[2,7,4,9,3],"target":[1]}' > /tmp/needle-head.json
bin/needle -mode train-head -head confidence \
  -model checkpoints/needle3.safetensors -input /tmp/needle-head.json \
  -steps 10 -lr 0.001 -out checkpoints/needle3-confidence-tuned.safetensors
```

For router training, use `-head router` with a class target such as `[2]`. For embedding training, supply a normalized vector of the checkpoint's actual embedding dimension and use `-head embedding`.

Add `-numerics needle3-cq4-a8-kv8` to train through the Needle 3 CQ/A8/KV8 straight-through reference. Omit it for FP32. The output remains an F32 safetensors checkpoint; it does not become a packed `.cact` archive. The CLI refuses to replace an existing `-out` path.

`-steps` repeats the supplied sequence with a fixed learning rate. This is not a shuffled dataset trainer with minibatches or an automatic train/validation split. The caller owns those choices. `-work-mib` bounds the logical per-call workspace; it is not a process RSS limit and excludes persistent optimizer state.

Read the trained head without treating it as calibrated:

```sh
bin/needle -mode confidence \
  -model checkpoints/needle3-confidence-tuned.safetensors \
  -input /tmp/needle-head.json
```

### Go API

```go
m, err := needle.Load("checkpoints/needle3.safetensors")
// Check err before using m.
opts := needle.Options{MaxWorkBytes: 512 << 20}
loss, gradients, err := m.HeadLossGrad(ids, needle.Confidence, []float32{1}, opts)
optimizer := needle.NewAdamW()
next, loss, err := m.TrainHeadStep(optimizer, ids, needle.Confidence, []float32{1}, 0.001, opts)
// Check each error. m remains unchanged; next owns the updated head parameters.
```

The explicit objectives above are also used by the JAX parity fixture. That checks the mathematics, not whether the labels or training recipe are useful for a real task.

## Needle 2 heads

Needle 2 uses different parameter names and pooling: `contrastive_head` has four probes, `confidence_head` eight, with one softmax over all token/layer cells. It does not use Needle 3's per-layer RMS and second query pool. Call `Head(ids, needle.Contrastive, opts)` or `Head(ids, needle.Confidence, opts)` on a source checkpoint that contains the corresponding tensors. `contrastive_dim` sets the output width (upstream default 128); normalized outputs use the same epsilon as upstream. Confidence returns raw logits, not calibrated probabilities.

For frozen-trunk training, `HeadLossGrad` and `TrainHeadStep` accept `Contrastive` with a unit-vector MSE target or `Confidence` with a BCE target. Contrastive MSE is **not** the upstream paired contrastive/InfoNCE objective. `contrastive_head/log_temp` is preserved as a scalar checkpoint parameter and does not participate in this objective; no zero-gradient optimizer update or weight decay is applied to it. There is no Needle 2 router or Needle 3-style `embedding` head. CQ options and deployment archives remain unsupported for these heads.

```sh
# tokens.json contains {"tokens":[2,7,4,9,3],"target":[1,0,0,0]}
# The source checkpoint must have a four-dimensional contrastive head.
bin/needle -model checkpoints/needle2-source.safetensors -input tokens.json \
  -mode train-head -head contrastive -steps 5 -lr 0.001 \
  -out checkpoints/needle2-contrastive-trained.safetensors
bin/needle -model checkpoints/needle2-contrastive-trained.safetensors \
  -input tokens.json -mode contrastive
```

Both commands use source safetensors and token IDs; they do not imply Needle 2 `.cact` loading or text tokenizer support. See the [Needle 2 validation record](../validation/needle2-heads-20260920.md) for pinned upstream outputs, padding masks, head gradients and native ARM checks.

## Half-width slicing (Needle 3)

`m.SliceWidth(width)` implements the pinned Needle 3 `width_config` and `width_slice` tensor selection, not arbitrary truncation. `cmd/needle -width N` applies the same operation before inference or training. If `-layers` is also specified, depth slicing runs first.

Admission requires a source Needle 3 model with a power-of-two parent width, nonempty `ladder_widths` indicating trained split permutations, and a requested width exactly half of the parent. Attention heads halve while KV heads stay fixed, so the new query-head count must still be divisible by the KV-head count. The parent/child Hadamard block geometry must share the same second factor size. Unsupported geometry fails; the code does not manufacture new weights.

Engram tables keep the first half of each order's heads, and hashing preserves the original seed stride. The child configuration must actually produce that halved geometry. Upstream does not halve an explicitly fixed `engram_heads` field, so incompatible explicit-head configurations are rejected. Width slicing of AB-scaled weights or deployed archives is unsupported. A child has no remaining width-ladder metadata; repeated arbitrary halving is not offered.

The released width-768 Needle 3 archive used in the [smoke tests][real] is **not** an admissible source for half-width slicing. It is both a deployment archive and non-power-of-two. Do not use `-width 384` on it and expect a trained smaller model.

For an actual width-16 source checkpoint trained with a width-8 rung, an invocation would be:

```sh
bin/needle -model checkpoints/needle3-width16-source.safetensors \
  -input /tmp/needle-head.json -width 8 -mode embedding
```

This names the tested geometry, not a distributed pretrained checkpoint. Slicing copies changed tensors and shares only immutable private tensors that remain unchanged; the public checkpoint API still returns caller-owned copies.

## Validation and limits

An engram-bearing source fixture checks the admissible 1,024-to-512 rung, retained parent hash seeds, FP32/CQ logits and heads, and cached decoding on Intel and native ARM; see the [engram-width validation record](../validation/needle-engram-width-20260920.md). Source CQ reference preparation uses pre-scaled dense Hadamard products to preserve upstream rounding behavior; deployed archive and packed execution are unchanged.

The [head/width validation report][validation] records FP32/CQ loss and gradient comparisons, exact tensor cuts, native ARM execution and allocation measurements. It also lists the combinations not yet qualified. The implementation does not claim a calibrated confidence head, production training throughput, or that every source checkpoint supports a width rung.

[real]: ../validation/needle-packed-real-model-20260920.md
[validation]: ../validation/needle-head-width-20260920.md
