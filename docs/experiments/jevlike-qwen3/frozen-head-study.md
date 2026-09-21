## The cached head did not learn a useful decision rule

The bounded frozen-head learning check failed. Three rank-64 native heads scored
**18.33%, 18.33% and 16.67%** on the same 60 validation questions used by the direct
instruction scorer, which scored 80%. Their mean, **17.78%**, is below the 23.17%
uniform-random baseline. A cosine diagnostic using mean context features and
pooled option features also scored 18.33%.

This is a negative result for the fixed 160-example, five-epoch recipe, not a
proof that larger or differently trained heads cannot work. The [predeclared
learning policy][policy] requires accuracy and meaningful evidence dependence
before expansion; neither held here. No larger cache, hyperparameter sweep or
LoRA run was started. Final-test data remains unread.

## Same questions, different architecture

The training sample contains 32 examples per source/config, selected by the same
hash rule within the training partition only. Validation uses the existing 60
originals and all 228 control requests; calibration uses the separate 60
originals. All were admitted without truncation. CLINC is the labelled 8-choice
including-OOS task, not full-intent accuracy.

The backbone is the same pinned instruction Qwen3-4B used in the direct study.
The head reads all final-normalised rows of `Evidence:\n...\nQuestion:\n...`
and independently mean-pooled candidate representations. The direct scorer
instead reads a joint chat-formatted prompt. Text and candidate sets match, but
these are intentionally different computation and formatting policies.

| Validation task, 12 originals each | Seed 7 | Seed 17 | Seed 27 | Direct |
|---|---:|---:|---:|---:|
| ARC Challenge | 33.33% | 16.67% | 25.00% | 91.67% |
| ARC Easy | 25.00% | 8.33% | 8.33% | 83.33% |
| CLINC 8-choice | 0.00% | 8.33% | 8.33% | 91.67% |
| CommonsenseQA | 0.00% | 25.00% | 16.67% | 66.67% |
| MultiNLI | 33.33% | 33.33% | 25.00% | 66.67% |

All three heads changed **zero of 60** selected answers on option reversal.
Explicit equivariance checks also passed **288 permutations per seed**, with zero
logit difference after stable-ID mapping and zero unique-argmax changes. Unit
tests cover all permutations through four choices, padded batches and both cache
dtypes. See the [candidate and permutation contract](candidate-contract.md).
This is expected from the architecture and does not rescue the answer quality. CLINC
showed zero normal-over-shuffled-evidence gain in every seed. NLI's gain was
+16.67, 0 and -8.33 percentage points. The expansion requirement was at least ten
points on both evidence tasks, in addition to accuracy above random plus ten.

The learning curves are retained in the [full results][results]. Loss was high
and did not show a stable useful fit; the recorded best validation epoch was
restored separately from the resumable current optimiser state. There was no
selection of a preferred seed by validation accuracy.

## Calibration cannot fix wrong answers

Each seed fitted one temperature on its separate calibration requests only. All
three reached the upper bound of **20** (within floating-point optimisation
precision), so those fits are boundary solutions, not well-determined scales.

| Normal validation | Raw NLL | Calibrated NLL | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|
| Seed 7 | 5.4584 | 1.5600 | 0.5100 | 0.0924 |
| Seed 17 | 3.4247 | 1.5140 | 0.4794 | 0.0835 |
| Seed 27 | 3.2407 | 1.5278 | 0.5281 | 0.0957 |

Accuracy did not change. Lower ECE here mostly reflects less confident guesses;
it does not make a below-random scorer useful. Per-task metrics, Brier scores,
reliability bins and risk/coverage intervals are retained. No deferral threshold
or deployment recommendation follows from them.

## Cache and resume checks that passed

The F32 feature cache occupies **106,630,654 bytes** for 1,159 unique context or
pooled-option records. Training extraction took 469.4 seconds after an
intentional one-request interruption, validation/control extraction 199.8
seconds and calibration 158.0 seconds. These include checkpoint rehashing and
local filesystem work; repeated options and reversed requests hit existing
entries. Backbone hashes were identical before and after every stage.

Entries bind the verified model-file digest, dataset manifest, text, exact token
IDs, backend/precision, pooling, dtype and token limits. Each binary record has
bounded shapes, payload and whole-record checksums; atomic publication refuses
to overwrite a completed entry. Missing offline entries fail instead of loading
a backbone. F32 and FP16 are different contracts, so a same-width or
wrong-representation checkpoint cannot load implicitly. An interrupted writer
retains completed records; stale lock removal requires checking that its owner
has stopped.

A separate FP16 conversion was an explicit ablation, not an adopted default.
Across 228 validation/control decisions using seed 7's head, maximum feature
delta was **0.03125**, maximum logit delta **0.0326862**, and **zero decisions
changed**. Combined F32 and FP16 storage was **160,228,348 bytes**, comfortably
inside the 12 GiB pilot-cache cap. That single-head comparison does not establish
an FP16 training result.

Native training ran with NVIDIA disabled, no encoder fallback, rank 64, batch
4, learning rate 0.002, clip norm 1 and seeds 7/17/27 for five epochs. It took
51.0, 82.1 and 47.2 seconds respectively; the first figure excludes the short
intentional interruption. CPU contention during extraction makes these timings
observations rather than controlled throughput benchmarks. Peak training RSS was
424, 362 and 327 million bytes approximately (the logs record KiB precisely).

The real seed-7 interrupted/resumed run and a separate uninterrupted run produced
**byte-identical optimiser-state and checkpoint files**. State includes current
and best parameters, Adam moments, step, epoch, cursor and partial epoch loss;
shuffle is reconstructed from seed+epoch. Gradient-norm reductions use sorted
parameter names so process-specific map order cannot perturb clipping.

## Cached latency is not fresh-input latency

Normal validation cache-read-plus-head p50 was 1.42--1.47 ms and p95
3.35--3.76 ms. Those numbers exclude backbone extraction. A fresh GPU run of the
first validation request matched its cached head logits exactly but took
**2.7572 seconds**. The new key-location smoke request took **2.3941 seconds**.
It stated that the amber key was in the blue drawer and that the red drawer was
empty; seed 7 chose the red drawer. The request was machine-authored and has not
been independently human-reviewed.

A separate [three-scenario online benchmark](candidate-contract.md) uses complete
candidate statements and distinguishes fresh extraction, reused candidate
features and fully cached RAM inputs. Their p50 latencies were 2.465 s, 0.801 s
and 0.690 ms, with exactly equal logits; the same wrong drawer was selected.
The original benchmark fragments remain an explicitly exploratory contract,
not retrospectively self-contained alternatives.

The serving model was restored healthy after extraction and fresh execution.
Free disk remained about 73 GiB. No second checkpoint was downloaded, and neither
InvokeAI nor ComfyUI was changed.

## Reproduction and what is still open

The extraction/training implementation is pinned in the run records to
`716f95c389a6f250126386ffad22decea19c9c48`; the later cache-publication guard
rejects invalid rounded features before writing, without changing valid feature
bytes. Use the prepared dataset and installed instruction assets from the
[direct study][direct]. All output paths below must be new for a fresh run.

```bash
ROOT=checkpoints/jevlike-qwen3
MODEL=$ROOT/Qwen--Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c
bun scripts/jevlike-direct-pilot.ts --partition train --per-task 32 \
  --output "$ROOT/head-train-v1"
go build -o bin/jevlike ./cmd/jevlike
# Repeat for direct-pilot-v1 and direct-calibration-v1 after head-train-v1.
bin/jevlike cache-extract -encoder-model "$MODEL" \
  -verified-assets "$ROOT/instruction-model-verified.json" \
  -dataset "$ROOT/pilot-v4" -selection "$ROOT/head-train-v1/selection.json" \
  -requests "$ROOT/head-train-v1/requests.jsonl" -cache "$ROOT/features-f32-v1"
# Use -plan before extraction; -max-new 1 exercises extraction resumption.
GO_PHERENCE_DISABLE_NVIDIA=1 bin/jevlike cached-train \
  -cache "$ROOT/features-f32-v1" -dataset "$ROOT/pilot-v4" \
  -train-study "$ROOT/head-train-v1" -validation-study "$ROOT/direct-pilot-v1" \
  -state "$ROOT/head-runs-v1/seed-7-state.json" \
  -checkpoint "$ROOT/head-runs-v1/seed-7.json" \
  -code-revision 716f95c389a6f250126386ffad22decea19c9c48 \
  -seed 7 -rank 64 -epochs 5 -batch-size 4 -learning-rate 0.002
# Use -max-steps 1 then repeat without it to test exact optimiser resume.
# Repeat for seeds 17 and 27, preserving every report.
bin/jevlike cached-score -cache "$ROOT/features-f32-v1" \
  -dataset "$ROOT/pilot-v4" -study "$ROOT/direct-pilot-v1" \
  -checkpoint "$ROOT/head-runs-v1/seed-7.json" \
  -output "$ROOT/head-runs-v1/seed-7-validation.jsonl"
```

`direct-report` consumes cached-score output using the same partition checks as
the direct baseline. `cached-similarity`, `cache-compare` and `feature-score`
provide the cosine, FP16 and fresh-input checks; their help lists required paths.
`feature-score` now requires an explicit candidate contract in new requests.
`head-permutation` and `head-online-bench` reproduce the added mapped-logit and
three-scenario checks.
Raw checkpoints, datasets and feature records stay outside Git; the published
report includes their hashes.

The whole issue is **not complete**. The conditional expansion gate stopped this
recipe. Independently human-reviewed new cases are still missing. At the time of
this head study, direct prefills were independent and serialised and the compact
encoder retained no per-layer KV prefix state; these measurements did not test
transformer-prefix reuse.

The later [native prefix study](prefix-study.md), explicitly included in issue #2
by the additional acceptance comment, implements and validates that execution
path without changing this head recipe or its quality results. There is still
no final-test result, LoRA justification or desktop-integration claim. Further
learning work needs a reviewed failure set and an explicit new recipe.

[policy]: frozen-head-policy.md
[results]: frozen-head-results.json
[direct]: direct-study.md
