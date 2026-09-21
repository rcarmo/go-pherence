## Frozen Qwen3 final evaluation

The frozen held-out evaluation completed all **1,440** test originals on 2026-09-21. Every arm admitted every row. The direct Qwen3-4B Instruction scorer reached **81.25%** accuracy; the three frozen heads reached **21.11%**, **22.85%** and **22.64%**, for a **22.20%** mean against the pooled **23.52%** uniform-random expectation.

These results close the approved experiment with a negative result for the trained heads. The direct scorer is a useful baseline, but its earlier option-order failure still applies. No final-test result changes the frozen development gates or promotes either method for deployment.

## Frozen scope and completion

The [policy](final-evaluation-policy.md) and [freeze](final-freeze.json) were committed before test requests were constructed. The run used:

* the normal variant only, with the original candidates and labels;
* the pinned Qwen3-4B Instruction checkpoint and no-thinking direct prompt;
* frozen heads for seeds 7, 17 and 27;
* F32 features and the existing feature contract;
* the previously fitted temperatures, applied only while reporting;
* no refit, prompt change, training, new seed, threshold selection or completed-row replay.

The first attempts had published 676 immutable records before GPU bus loss. The authorised recovery resumed from the first absent file, row 677, and wrote the remaining 764 records. `final-eval` then rechecked the model assets and the 16-file freeze. `final-report` validated all record/request identities before reading outcomes.

| Identity | SHA-256 |
|---|---|
| Freeze | `229fb1ec535d8f23205dc1fc17f01234115ae97c32b70f1bfff3fa071afcd527` |
| Requests | `f2b54206ea67e0983b60886d8fe4c8a2ac6a96dc020fdd7055f131e90d3e36a4` |
| Dataset manifest | `459d5c1f6df494844e1aa21c485df574d6ee315420f00d8da70adf0f8efdefa1` |
| Test partition | `fe235bc69077cca346bed6344fe9b8bfff7a3e0da8730fed9dce8550b4b5d27a` |
| Ordered record-hash manifest | `bd318c460f1d674af3ac2d585d1e8d2bd89bb1cf8f419be7aac341ac921a7b0f` |
| Generated full report | `5eb1bda156b23fc9f9991f564c270055d5346a146f5935322941e4979bd87aec` |

The compact [machine-readable results](final-results.json) retain all pooled, per-task and choice-count scalar metrics. The ignored checkpoint directory retains the 1,440 records, four result JSONLs and full reliability/risk-coverage arrays.

## Pooled results

All rows belong to the common admitted cohort, so admitted-only accuracy and accuracy with rejections counted as wrong are identical. Temperature scaling changes NLL, summed multiclass Brier score and ten-bin expected calibration error (ECE), but never the selected answer.

| Arm | Frozen T | Requested / admitted / rejected | Accuracy | Random | NLL raw / calibrated | Brier raw / calibrated | ECE raw / calibrated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Instruction | 6.5667 | 1,440 / 1,440 / 0 | 81.25% | 23.52% | 2.410 / 0.550 | 0.357 / 0.275 | 0.171 / 0.046 |
| Head seed 7 | 20.0000 | 1,440 / 1,440 / 0 | 21.11% | 23.52% | 5.669 / 1.552 | 1.164 / 0.779 | 0.474 / 0.073 |
| Head seed 17 | 20.0000 | 1,440 / 1,440 / 0 | 22.85% | 23.52% | 3.275 / 1.508 | 1.088 / 0.768 | 0.446 / 0.039 |
| Head seed 27 | 20.0000 | 1,440 / 1,440 / 0 | 22.64% | 23.52% | 2.943 / 1.506 | 1.100 / 0.767 | 0.448 / 0.037 |

The direct scorer exceeds random expectation on every task. MultiNLI is its weakest task at 66.25%. The heads cluster around random: individual seeds exceed expectation on some tasks and fall below it on others, while their pooled mean is 1.32 percentage points below random. Seed 7 reaches only 2.19% on the prepared eight-choice CLINC task.

The low calibrated head ECE values need the accuracy figures beside them. Each frozen head temperature hit the permitted upper boundary near 20, which flattens poor predictions towards the offered-set distribution. Calibration improves reported probabilities; it does not recover decision quality.

## Per-task accuracy

CLINC is the prepared labelled eight-choice task, including OOS. It is not full 151-intent deployment accuracy. Random values differ slightly within the ARC groups because a few source examples have a non-standard number of retained choices.

| Arm | Task | Examples | Accuracy | Random |
|---|---|---:|---:|---:|
| Instruction | ARC Challenge | 320 | 85.94% | 25.05% |
| Instruction | ARC Easy | 320 | 92.19% | 24.97% |
| Instruction | CLINC OOS `plus` | 320 | 85.31% | 12.50% |
| Instruction | CommonsenseQA | 160 | 71.88% | 20.00% |
| Instruction | MultiNLI | 320 | 66.25% | 33.33% |
| Head seed 7 | ARC Challenge | 320 | 24.69% | 25.05% |
| Head seed 7 | ARC Easy | 320 | 23.44% | 24.97% |
| Head seed 7 | CLINC OOS `plus` | 320 | 2.19% | 12.50% |
| Head seed 7 | CommonsenseQA | 160 | 22.50% | 20.00% |
| Head seed 7 | MultiNLI | 320 | 33.44% | 33.33% |
| Head seed 17 | ARC Challenge | 320 | 26.56% | 25.05% |
| Head seed 17 | ARC Easy | 320 | 20.94% | 24.97% |
| Head seed 17 | CLINC OOS `plus` | 320 | 10.63% | 12.50% |
| Head seed 17 | CommonsenseQA | 160 | 19.38% | 20.00% |
| Head seed 17 | MultiNLI | 320 | 35.00% | 33.33% |
| Head seed 27 | ARC Challenge | 320 | 22.50% | 25.05% |
| Head seed 27 | ARC Easy | 320 | 20.63% | 24.97% |
| Head seed 27 | CLINC OOS `plus` | 320 | 11.56% | 12.50% |
| Head seed 27 | CommonsenseQA | 160 | 24.38% | 20.00% |
| Head seed 27 | MultiNLI | 320 | 35.00% | 33.33% |

The frozen temperatures reduce NLL, Brier and ECE on each task without changing the accuracies above:

| Arm | Task | NLL raw / calibrated | Brier raw / calibrated | ECE raw / calibrated |
|---|---|---:|---:|---:|
| Instruction | ARC Challenge | 1.900 / 0.468 | 0.271 / 0.228 | 0.135 / 0.047 |
| Instruction | ARC Easy | 0.755 / 0.237 | 0.140 / 0.107 | 0.071 / 0.052 |
| Instruction | CLINC OOS `plus` | 2.233 / 0.525 | 0.278 / 0.226 | 0.134 / 0.052 |
| Instruction | CommonsenseQA | 3.561 / 0.802 | 0.542 / 0.401 | 0.268 / 0.099 |
| Instruction | MultiNLI | 4.175 / 0.842 | 0.645 / 0.474 | 0.317 / 0.171 |
| Head seed 7 | ARC Challenge | 1.998 / 1.386 | 0.899 / 0.751 | 0.281 / 0.036 |
| Head seed 7 | ARC Easy | 2.113 / 1.394 | 0.945 / 0.753 | 0.303 / 0.033 |
| Head seed 7 | CLINC OOS `plus` | 16.952 / 2.299 | 1.895 / 0.937 | 0.939 / 0.216 |
| Head seed 7 | CommonsenseQA | 2.941 / 1.600 | 1.040 / 0.796 | 0.374 / 0.011 |
| Head seed 7 | MultiNLI | 2.976 / 1.107 | 0.978 / 0.668 | 0.428 / 0.056 |
| Head seed 17 | ARC Challenge | 2.569 / 1.385 | 0.980 / 0.749 | 0.387 / 0.025 |
| Head seed 17 | ARC Easy | 3.377 / 1.405 | 1.164 / 0.759 | 0.508 / 0.077 |
| Head seed 17 | CLINC OOS `plus` | 3.526 / 2.070 | 1.245 / 0.873 | 0.509 / 0.041 |
| Head seed 17 | CommonsenseQA | 6.146 / 1.643 | 1.257 / 0.812 | 0.562 / 0.067 |
| Head seed 17 | MultiNLI | 2.190 / 1.103 | 0.878 / 0.670 | 0.323 / 0.013 |
| Head seed 27 | ARC Challenge | 2.075 / 1.389 | 0.986 / 0.752 | 0.348 / 0.044 |
| Head seed 27 | ARC Easy | 2.277 / 1.398 | 1.010 / 0.755 | 0.383 / 0.063 |
| Head seed 27 | CLINC OOS `plus` | 4.424 / 2.087 | 1.286 / 0.877 | 0.532 / 0.036 |
| Head seed 27 | CommonsenseQA | 3.130 / 1.606 | 1.101 / 0.798 | 0.455 / 0.013 |
| Head seed 27 | MultiNLI | 2.903 / 1.102 | 1.119 / 0.669 | 0.531 / 0.032 |

[final-results.json](final-results.json) also records the choice-count groups required to interpret the mixed pooled random baseline.

## Runtime and GPU record

The missing-row run used the approved NVIDIA GeForce RTX 3060 with driver 580.173.02. A five-second watchdog stopped the process on a new NVIDIA Xid, a failed device query or a temperature of 83 C. None of those conditions occurred.

| Measurement | Value |
|---|---:|
| Newly evaluated originals | 764 |
| Resume wall time | 2,547.687 s (42 min 27.687 s) |
| Final F32 cache size | 609,520,817 bytes |
| Existing F16 cache size | 53,597,694 bytes |
| Active telemetry samples | 475 |
| Active temperature, minimum / mean / maximum | 65 / 77.823 / 81 C |
| Active power, mean / maximum | 130.134 / 137.090 W |
| Mean active GPU utilisation | 98.396% |
| Peak device memory | 7,399 MiB |
| Xids / device-query failures | 0 / 0 |
| Software / hardware thermal slowdown | 0 / 0 us |

The full [telemetry CSV](final-gpu-telemetry.csv) has 478 samples including load and shutdown. Its SHA-256 is `a4bdf466c2917328451e5594dd0d38e5f416b75701fe262da779494ed715af58`. The GPU returned to P8 idle state at 52 C, 14.40 W and 33 MiB after the evaluator closed.

Stored per-record timing covers work from both the interrupted and resumed attempts. Across all 1,440 records, direct decisions total 2,908.600 s and feature extraction totals 3,160.763 s. The three cached-head forwards total 2.672 s, 2.566 s and 2.507 s. Those head totals exclude backbone feature extraction and must not be compared with direct GPU time as equivalent end-to-end latency.

## Interpretation and limits

The final test agrees with the development decision: the fixed rank-64 attention-head recipe did not learn a useful general scorer. Its mean final accuracy is below random and the three seeds vary materially by task. The training expansion gate remains failed.

The direct Instruction arm is much stronger, and frozen temperature scaling brings its pooled NLL from 2.410 to 0.550 and ECE from 0.171 to 0.046. Its validation controls had already changed 8 of 60 answers when options were reversed, above the predeclared 10% limit. The final normal-only pass does not retest or erase that failure.

The approved closure deferred independent human-reviewed cases and the wider transfer/stress matrix. The 24 machine-authored cases remain unreviewed and unscored. No confidence-based deferral threshold, production service behaviour, full-intent CLINC accuracy or separate relevance reranker was evaluated. Future model or prompt changes require a new selection protocol; these test outputs cannot become tuning data for the closed experiment.

## Reproduction

Run from the repository root with the frozen local assets present. The output directories must already contain only the immutable records being resumed, or must not exist for a new report.

```bash
go build -o /tmp/jevlike-final ./cmd/jevlike

/tmp/jevlike-final final-eval \
  -freeze docs/experiments/jevlike-qwen3/final-freeze.json \
  -study checkpoints/jevlike-qwen3/final-once-v1 \
  -encoder-model checkpoints/jevlike-qwen3/Qwen--Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c \
  -cache checkpoints/jevlike-qwen3/features-f32-v1 \
  -other-cache checkpoints/jevlike-qwen3/features-f16-v1 \
  -output checkpoints/jevlike-qwen3/final-once-v1/records \
  -plan

/tmp/jevlike-final final-report \
  -freeze docs/experiments/jevlike-qwen3/final-freeze.json \
  -study checkpoints/jevlike-qwen3/final-once-v1 \
  -records checkpoints/jevlike-qwen3/final-once-v1/records \
  -output checkpoints/jevlike-qwen3/final-report-v1
```

Remove `-plan` only for an authorised frozen execution. `final-eval` validates and skips existing records; it does not replace them. `final-report` refuses an incomplete or locked record set and applies the frozen calibrations without refitting.
