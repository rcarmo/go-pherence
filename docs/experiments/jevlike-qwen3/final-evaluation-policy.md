## Final held-out evaluation and research closure

On 2026-09-19 the owner approved using the existing 1,440 held-out labelled
examples once, then closing issue #2 as a completed experiment. Fresh
human-reviewed cases and the wider transfer/stress-test matrix are explicitly
deferred, not reported as performed. Neither scorer receives a deployment or
production-readiness endorsement from closure.

Before reading `test.jsonl`, freeze the installed Instruction checkpoint, all
three already-trained heads (seeds 7, 17 and 27), the exact prompt/feature
renderers, candidate sets, tokenizer/feature contracts, and the existing
calibration report hashes. The freeze record uses the test-artifact hash already
published in the prepared dataset manifest; creating it does not inspect test
rows. Commit and publish the policy/record before constructing test requests.

Evaluate all 1,440 originals with their original candidates and labels, in a
deterministic order. Use only the normal variant. This is one final pass, not a
new permutation/control sweep or a model-selection exercise. The fixed
validation controls and their failures remain part of the conclusions. The Base
checkpoint is not refetched: the final comparison is the selected Instruction
direct scorer versus each of the three frozen heads on that same backbone.

Use the existing no-thinking direct prompt and 512-token limit, plain head
context formatting, 512-token context/128-token option limits and F32 features.
Temperature 1 produces raw scores. Apply each previously fitted calibration
parameter only in reporting: Instruction T=6.5666879722, and the stored values
near the upper bound 20 for the heads. No refitting, new seed, prompt revision,
training, rank/learning-rate change or confidence/deferral threshold selection.

Admission rejection is a result, not permission to truncate. Record all
originals and each arm's explicit outcome. Infrastructure/decoder/GPU/cache
corruption failures stop the run instead of being counted as ordinary model
rejections. Report each arm's admitted sample and the common admitted cohort;
include requested/admitted/rejected counts, rejection causes and accuracy with
rejections counted as incorrect as well as admitted-only metrics. On identical
admitted candidates report pooled/per-task accuracy, uniform random baseline,
NLL, summed Brier and ten-bin ECE, raw and frozen-calibrated. Report all seeds
and their mean without choosing a preferred head after looking at test results.

Retain raw selected logits, candidate IDs, timings and checksummed provenance.
One atomic immutable record contains the complete set of four arm outcomes for
each original. Restart may resume only missing records after revalidating the
freeze and request identities; completed decisions cannot be overwritten or
rescored. A computation interrupted before atomic publication may be repeated,
but the failed attempt must be retained in the run log. This is crash recovery,
not another statistical pass. Report timing totals as observed work; do not
compare GPU-containing direct latency with cached-head CPU latency as if they
had the same cost boundary.

The GPU encoder stays resident for the run. Shared immutable feature records
avoid repeated backbone extraction across heads. Model assets remain at most
12 GiB, combined F32/FP16 feature caches at most 12 GiB, with at least 30 GiB
free disk; refuse execution if the preflight upper bound cannot fit. Restore the
previous serving-model state after execution, including failures.

Close #2 only after all expected originals have complete records, freeze and
checkpoint hashes still match, reports include failures and the final scope
limitations, tests pass and the report is pushed. A bad final result is a valid
research outcome. Further improvements require a new experiment with a new
selection protocol; these final-test outputs must not become tuning data under
the existing issue.
