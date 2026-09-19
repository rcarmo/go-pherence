## Direct calibration and promotion policy

The 60-example validation study can expose failures, but it is too small to
establish a deployment error bound. Keep its final-test partition closed.

The instruction checkpoint is the only new direct-scoring configuration in this
stage: the same prompt, code mapping, 512-token limit and exact 228 requests as
Base, with no prompt search. Before inspecting its completed validation report,
use these provisional requirements for a larger validation comparison:

* Normal pooled accuracy must reach 80%, option reversal must change no more
  than 10% of answers, and NLI/CLINC normal accuracy must exceed their matched
  shuffled-evidence accuracy by at least 10 percentage points each. A failed
  requirement is reported as a failure, not adjusted after looking at results.
* Passing allows further measurement, not deployment. Compare on the full
  validation set before opening final test; record uncertainty and each task's
  sample size. CLINC means the prepared labelled 8-choice task, including OOS.
* Use a fixed calibration-only sample of 12 examples per source/config, selected
  by the same hash rule within the separate calibration partition. Fit one
  temperature to **normal requests only**, minimising NLL within [0.05,20].
  Reversed and evidence-control requests do not contribute to the fit. This is a
  small calibration pilot; boundary solutions and per-task degradation must be
  reported, not hidden by pooled metrics.
* Freeze the parameter with model-file-set identity, dataset-manifest hash,
  request hash and scored-result hash. Applying it to validation is a diagnostic;
  the fitted calibration score is not an independent quality result. No deferral
  threshold will be promoted from these 60 calibration examples.

The report uses ten equal-width reliability bins, summed multiclass Brier,
uniform-choice random and first-position baselines, and Wilson 95% intervals for
error at distinct confidence thresholds. Equal confidence enters coverage as a
group. Option-order comparisons and evidence-control accuracy use the common
admitted examples; rejection counts are always retained. Latency percentiles use
the nearest-rank convention.

A training cache or frozen-head comparison must answer a concrete failure or
cost question exposed by these measurements. Do not generate the full pilot
cache merely because the extractor fits. Three-seed head results, fresh
human-reviewed cases and exact-prefix reuse remain separate work; none is
implied by a direct-scoring numerical pass.

[Study selection policy](direct-study-policy.md) | [Experiment status](README.md)
