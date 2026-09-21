# Direct-score validation policy

This policy is defined before viewing the complete Base validation results.
It is an exploratory baseline gate, not final-test acceptance.

* Select 12 original validation examples per source/config by lowest
  SHA256 of `direct-pilot-v1:source_id`: ARC Easy, ARC Challenge, CommonsenseQA,
  CLINC `plus`, and MultiNLI. This gives 60 originals and 228 requests.
* Preserve the original grouped partitions and candidate sets. Never read final
  test for prompt/model selection. Reject overlength rendered prompts; report
  rejection counts and evaluate matched admitted examples across variants.
* For every row run normal, reversed-option and option-only forms. NLI/CLINC
  additionally get no-evidence and a different example's evidence from the same
  task. Knowledge QA has no separable evidence field, so do not label a different
  question as meaningful shuffled evidence.
* Use temperature 1 for this validation study. Fit probabilities only on the
  calibration partition after choosing the configuration. Report accuracy, NLL,
  summed multiclass Brier, ten-bin ECE, tied-threshold risk/coverage with Wilson
  error intervals, uniform random and first-option baselines. First-option is a
  position diagnostic, not a learned majority-class predictor.
* Investigate an instruction checkpoint if pooled normal accuracy is no better
  than random plus 10 percentage points, option-order changes exceed 10%, or
  NLI/CLINC shows no meaningful gain over evidence controls. Small per-task
  counts cannot support precise superiority claims; retain all negative results.
* Select neither prompts nor thresholds using final-test outputs. Freeze the
  next evaluation policy after validation and before calibration/final testing.

The instruction candidate is `Qwen/Qwen3-4B` at
`1cfa9a7208912126459214e8b04321603b3df60c`, pinned in `sources.json`.
Its chat-template hash differs from Base and needs independent rendering/logit
fixtures. No additional architecture compatibility follows from its name.

Two full 4B checkpoints exceed the approved 12 GiB model-asset budget. After Base
validation, preserve its verified manifest, references, per-row results and
source pins, then remove only its reproducible safetensor shards before the
instruction download. Do not delete its reference results or silently repoint
Base cache/checkpoint identities. The downloader checks combined weight storage,
including partial files. Any later Base rerun requires the same sequential swap
or an explicit budget increase.

[Experiment status](README.md) | [Compact/direct numerics](compact-direct-validation.md)
