# Needle 2 auxiliary heads -- 2026-09-20

Needle 2's FP32 contrastive and confidence heads now work through `Head`, the frozen-trunk training APIs and `cmd/needle`. The implementation follows the pinned version-2 pooling rather than substituting Needle 3's head design. The [head guide](../guides/needle-head-training.md) has API and CLI examples and checkpoint restrictions.

## Contracts kept separate

At upstream `741ee892c5f8c4f5c0bb467c9566ea7a1eba919b`, the contrastive head uses four probes and the confidence head eight. Both softmax over flattened token/layer cells, excluding padding tokens; neither has Needle 3's per-layer RMS or second query pool. Contrastive output is L2-normalized with the upstream epsilon. Confidence returns a raw scalar logit. An all-padding input is rejected before softmax.

The public head names are `Contrastive` (`-mode contrastive`) and `Confidence`. Needle 2 rejects the Needle 3 `Embedding`/`Router` heads and CQ options. Its source checkpoint must contain the required geometry; missing or malformed tensors fail explicitly. `contrastive_dim` defaults to upstream's 128 if absent and is capped at 4,096 for head admission. No new archive or tokenizer compatibility is implied.

`HeadLossGrad` and `TrainHeadStep` support confidence BCE and normalized-vector MSE for the contrastive output, with the language-model trunk frozen. MSE is a supervised embedding primitive, **not** upstream's paired contrastive/InfoNCE training. The scalar `contrastive_head/log_temp` remains checkpoint-owned and is preserved through training/save/reload, but does not enter MSE and receives no optimizer update. The tests verify its zero upstream gradient for this objective. No calibration or dataset-level quality claim is made.

## Numerical evidence

[`scripts/needle2-head-reference.py`](../../scripts/needle2-head-reference.py) loads architecture and quantization code from the exact Git pin using `git show`, then runs JAX on CPU. It extends the existing tiny Needle 2 trunk fixture with deterministic seeded upstream heads, a four-dimensional contrastive output and explicit BCE/MSE targets. It emits all tensor shapes, outputs, padded-token outputs, losses and gradients to [`needle2-heads.json`](../../model/needle/testdata/needle2-heads.json).

```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/cpu-jax-venv/bin/python scripts/needle2-head-reference.py \
  --upstream /path/to/needle --output model/needle/testdata/needle2-heads.json

GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/needle \
  -run 'TestNeedle2Head' -count=1 -v
```

Tests compare both head outputs, padding-aware outputs, loss and every active head gradient using the existing head tolerances. They also check loss reduction after AdamW, unchanged trunk/other-head/temperature tensors, exact save/reload output, missing/invalid heads, cross-generation rejection, all-padding input and non-unit contrastive targets. CLI coverage trains the contrastive head for two steps, reloads it and checks uncalibrated output metadata.

Existing Needle 3 head tests still pass after extracting shared output normalization. The version-2 pooling remains separate. There is no tolerance increase.

## Execution gates

* Intel affected-package race tests and CPU-feature-disabled checks passed.
* Native ARM64 CIX P1, under `GOMAXPROCS=2 nice -n 10`: **110 model tests/subtests, 48 loader tests/subtests and nine CLI tests passed**. These are native test executions, not just a successful cross-build.
* Whole-tree `GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s`: **114 packages passed, 54 without tests**, confirmed exit zero.
* `go vet ./...`, `go build ./...` and Linux/RISC-V cross-build passed. RISC-V execution is not claimed.

`make docs-check` passed layout, diagrams, link-checker and docs tests: **364 Markdown files, zero broken links**. All **16 freeze-manifest entries**, both preserved evaluation binaries and **676 record hashes** matched their saved values; the evaluation remains frozen. No GPU or service activity was required. This is bounded fixture-level FP32 qualification, not released Needle 2 head quality or completion of the wider implementation queue.
