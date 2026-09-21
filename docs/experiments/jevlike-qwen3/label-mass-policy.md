# Offline allowed-label-mass diagnostic policy

This diagnostic is a bounded offline probe for the pinned direct-scoring Qwen3
instruction checkpoint. It is not a new validation set, not a dataset run, and
not a deferral or deployment rule.

## Fixed scope

Freeze the probe set before execution:

* Use **exactly** `model/jevlike/testdata/direct_questions.json`.
* This means the existing five synthetic direct-reference prompts only.
* Do not substitute pilot data, calibration data, or any final-test input.
* Compare against the matching output from `scripts/jevlike-direct-reference.py`
  for the same repository/revision so rendered prompt text, prompt token IDs,
  candidate token IDs and selected candidate logits are all checked first.

The direct fixture remains the admission check. The new label-mass output adds a
full-vocabulary normaliser diagnostic on top of that already-pinned subset.

## Arithmetic and implementation contract

`scripts/jevlike-label-mass.py` is CPU only:

* local Transformers load only, with `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1` and `local_files_only=True`
* float32 Qwen3 weights in memory from the pinned local BF16 checkpoint
* no sampling, no generation, no decoded continuation
* one last hidden state at a time, then one full-vocabulary output-head pass for
  that hidden state
* finite complete vocabulary logits required; any nonfinite value is a failure
* stable float64 `logsumexp` for the complete-vocabulary normaliser and the
  allowed-label numerator
* exclusive output creation and atomic rename; existing output or `.part` files
  are refused so prior failures are preserved for inspection

The script also checks the existing experiment storage constraints before load:

* the verified model file set must stay within the 12 GiB asset budget
* the model filesystem must have at least 30 GiB free
* no implicit fetch is allowed

Where possible, the script verifies the local model directory against
`docs/experiments/jevlike-qwen3/sources.json` and computes the same basename +
bytes + SHA256 file-set digest shape used by `cmd/jevlike score`.

## Reported quantities

For each of the five fixed prompts, compute:

* `log_allowed_mass`: `log(sum(exp(allowed_logits))) - log(sum(exp(full_vocab_logits)))`
* `allowed_mass`: `exp(log_allowed_mass)`
* conditional softmax over only the allowed answer-code tokens
* conditional entropy in nats over that allowed-token distribution
* conditional confidence = maximum conditional probability
* descriptive boolean `confidence >= 0.9 && allowed_mass < 0.1`

That boolean is **not** a promoted abstention rule. It is a predeclared label for
"the model strongly prefers one allowed token even though the allowed set gets
little total next-token mass".

The script also reports:

* exact prompt/render/token agreement versus the direct fixture
* maximum absolute selected-logit difference, with the unchanged `0.005` gate
* per-candidate conditional-probability differences versus the direct fixture
  logits
* Torch/Transformers versions, load time, full-head time, peak RSS and produced
  vocabulary-logit byte counts
* source-manifest hash, config hash and verified local file-set digest/model ID

## Descriptive bins

These bins are fixed before execution and are only for report slicing:

* allowed-mass bins: `[0,1e-6)`, `[1e-6,1e-4)`, `[1e-4,1e-2)`, `[1e-2,1e-1)`,
  `[1e-1,5e-1)`, `[5e-1,9e-1)`, `[9e-1,1]`
* confidence bins: `[0,0.5)`, `[0.5,0.75)`, `[0.75,0.9)`, `[0.9,0.99)`,
  `[0.99,1]`

Do not tune thresholds after looking at the five prompts and do not reinterpret
these bins as a service policy.

## Reproduction

Run from the repository root with the already-installed pinned instruction
checkpoint:

```bash
MODEL=checkpoints/jevlike-qwen3/Qwen--Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c
.venv-speaker/bin/python scripts/jevlike-direct-reference.py \
  --model "$MODEL" --repository Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --output checkpoints/jevlike-qwen3/reference/direct-instruction-v1.json
.venv-speaker/bin/python scripts/jevlike-label-mass.py \
  --model "$MODEL" --repository Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --reference checkpoints/jevlike-qwen3/reference/direct-instruction-v1.json \
  --output checkpoints/jevlike-qwen3/reference/label-mass-instruction-v1.json
```

The output file must be new. This probe is an offline numerical diagnostic for
issue #2, not a replacement for the direct-study policy, calibration policy or
held-out evaluation.

[Direct-score validation policy](direct-study-policy.md) |
[Direct calibration and promotion policy](direct-calibration-policy.md) |
[Compact/direct numerics](compact-direct-validation.md)
