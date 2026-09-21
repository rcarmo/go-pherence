## Native K/V prefix reuse, with isolated questions

Exact transformer-prefix reuse now matches independent native scoring on the
bounded development probes. Retaining 26 identical prompt tokens cut six-question
p50 latency from **11.118 s** to **8.235 s** with serial suffixes and **6.334 s**
with packed suffix projections. Logits and candidate-conditional probabilities
were identical in these measurements. Prefix construction cost **0.847 s** and
11,862,016 additional device bytes, including reusable branch scratch.

This implements the remaining native execution checks from [the acceptance
update][update]; it does not rerun or rehabilitate the failed head recipe. Direct
Instruction's 8/60 answer changes on option reversal also remain a promotion
failure. No training, dataset or asset expansion occurred. Final-test data was
not read.

## What is shared, and what is not

`FrozenPrefix` is owned by one compact GPU encoder. It stores immutable post-RoPE
keys and values for every layer, produced from an exact token prefix. The caller
cannot change its device pointers or contents. Only one prefix can be live per
encoder; it must be closed before replacement, and encoder close releases it.

The main group contains the five existing synthetic direct-reference prompts
plus one contradictory-instruction sibling. Their longest common prefix is
26 tokens, entirely before differing evidence/question text. A separate pair
with identical evidence shares 38 tokens and also passes. Each full prompt is
rendered/tokenised independently before splitting; no sibling question or
instruction is appended to the retained state. Results carry stable question
and candidate IDs.

Serial execution reads the retained K/V separately for each suffix. Packed
execution concatenates *projection rows*, not attention contexts: each branch
has independent absolute RoPE positions and sees only prefix keys plus its own
causal suffix. Reusable K/V scratch is populated from immutable prefix buffers
for each branch. There are no padded projection rows; explicit valid-token
lengths must match ragged slices, and last-real-token selection is checked.

The tests compare each question alone, alongside irrelevant/contradictory
siblings, in reverse question order, and after concurrent calls, rejection and
interruption. Maximum/RMS selected-logit differences and probability movement
are all **zero** for this cohort. The predeclared limits remain max 0.005,
RMS 0.0002 and probability movement 0.001. The smallest reference top-two gap
was 8.29095, so these are not real-model near-tie observations. Exact and one-ULP
ties have separate arithmetic tests; no wider numerical tolerance was inferred
from them.

Two repeated combined GPU runs preserved the independent Transformers direct
and 401-token hidden-state gates. Native direct-reference maximum logit error
remains 6.49e-5; hidden maximum 0.00447845/RMS 4.19e-5. Prefix checksums were
unchanged after valid, invalid, reordered, concurrent and cancelled suffix work.
An interrupted prefix build left no live handle; rebuilding produced the same
checksum. GPU free memory returned to its original value after close.

## Bounded work and cancellation

Per call: at most eight branches, 512 aggregate real suffix tokens, 2,048
prefix-plus-suffix tokens and 128 selected candidate tokens. In-flight plus
queued requests are limited to 32 branches, 2,048 suffix tokens, 4,096 effective
tokens and 512 candidates. Tests exceed each dimension independently. Excess
work is rejected before allocating request copies or doing GPU work, rather than
bounding only the number of requests.

Prefix calls are safely serialised. Queue waits honour cancellation; active
calls check it between layers and drain submitted CUDA work before returning.
This is cooperative cancellation, not GPU kernel preemption. A cancelled call
returns no partial result and releases admission accounting. Tests cover mixed
lengths/candidate counts, timeouts during prefix construction and suffix work,
queued cancellation, rejection followed by valid work, and cache reuse after
interruption. The caller must keep its input slices unchanged for the duration
of a call. Low-level independent extraction APIs remain synchronous; the bounded
queue described here is the new prefix-scoring API, not a network service.

## Terminal work and measured cost

Production execution keeps BF16 transformer matrices and bounded F32 projection
scratch. It performs no sampling, decoding, continuation or vocabulary-wide
projection. Selected output-head rows use the same explicit CPU F64 dot products.
The audit removed final normalisation of unused rows from independent terminal
scoring; extraction still returns all rows when requested. Prefix scoring
normalises/downloads only each branch's last real row, and prefix construction
performs no output-head projection.

Five warm repetitions per cell, identical prompt IDs and candidate sets:

| Questions | Path | p50 | p95 | Questions/s at p50 |
|---:|---|---:|---:|---:|
| 1 | Fresh independent | 1.674 s | 1.745 s | 0.597 |
| 1 | Cached serial suffix | 1.195 s | 1.234 s | 0.837 |
| 1 | Cached packed suffix | 1.180 s | 1.202 s | 0.848 |
| 6 | Fresh independent | 11.118 s | 11.351 s | 0.540 |
| 6 | Cached serial suffixes | 8.235 s | 8.455 s | 0.729 |
| 6 | Cached packed suffixes | 6.334 s | 6.461 s | 0.947 |

The six-question numbers are complete-group latency, not per-question latency.
Packing matrix work is not evidence of parallel API requests or concurrent GPU
streams. Weights and prefix are already resident; initial hashing, model load and
prefix construction are reported separately. The retained prefix stays allocated
also during the fresh comparator, avoiding different allocator conditions.
Tokenisation is included in every timed call. OS file cache is warm; five samples
do not establish production tail latency.

The [machine-readable report][results] includes every observation, tokenisation,
upload, prefill/suffix, selected projection and download times, transfer bytes,
device-copy bytes, and memory readings. Prefix fingerprint downloads are explicit
diagnostic work outside timed requests. The additional fixed device allocation
left **4,488,298,496 bytes** free; no branch-time device allocation is made, so the
owned peak is the base 7,464,810,496 bytes plus 11,862,016 prefix/workspace bytes.
Driver/module overhead is reflected in observed free-memory readings rather than
mislabelled as encoder-owned memory. Closing restored **12,107,251,712 bytes** free.

## A confident conditional answer can have little label mass

The [offline diagnostic policy][mass] fixes the subset to the same five existing
reference prompts. A local CPU F32 Transformers pass computes the last hidden
row, selected logits and one full-vocabulary output vector. The production Go
selected-row path is unchanged; this normaliser cannot be called through it.

All selected logits match the independent reference exactly. Four probes assign
nearly all full-vocabulary probability to allowed answer tokens. On the fifth,
the winning conditional probability is **0.9997493**, but the total probability
of either allowed label is only **0.0000112710** (log mass -11.3933). Conditional
entropy is 0.00232946 nats. That distinction is the result, not a confidence or
deferral recommendation.

The predeclared descriptive bin was conditional confidence >=0.9 with allowed
mass <0.1. It flags one of five probes. It is not fitted to a dataset and is not
an action-authority rule. Repeated-run numerical equality, entropy, label mass
and calibrated correctness are separate quantities; the existing calibration
and quality failures are not superseded.

Full-head projection across all five cases took **0.218 s**, and F64 conversion/
Python log-normalisation another **0.212 s**. One F32 vocabulary output occupies
607,744 bytes; F64 conversion and Python-float working memory are separately
identified and included in process peak RSS **24,109,568 KiB**. This is CPU-only
reference arithmetic, not an added GPU memory or runtime cost. No checkpoint was
downloaded; the file-set digest and resource-floor checks match the local study.

## Reproduction and the remaining human step

```bash
MODEL=checkpoints/jevlike-qwen3/Qwen--Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c
JEVLIKE_QWEN3_MODEL_DIR="$PWD/$MODEL" JEVLIKE_PREFIX_TEST=1 \
  go test ./model/jevlike -run '^TestQwen3PrefixIsolation$' -count=2 -v

go run ./cmd/jevlike prefix-bench -encoder-model "$MODEL" \
  -verified-assets checkpoints/jevlike-qwen3/instruction-model-verified.json \
  -repeats 5

.venv-speaker/bin/python scripts/jevlike-label-mass.py --model "$MODEL" \
  --repository Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --reference checkpoints/jevlike-qwen3/reference/direct-instruction-v1.json \
  --output checkpoints/jevlike-qwen3/reference/label-mass-new.json

bun scripts/jevlike-review-packet.ts --output /workspace/tmp/jevlike-human-review-new
```

GPU runs require the previously authorised serving workload to be stopped, then
restored to its prior state. The recorded run restored it healthy. Disk free
space stayed about 73 GiB, and model/cache budgets were unchanged.

The review generator supplies **24 unscored, machine-authored mechanics cases**
with proposed labels, family IDs and pending-review fields. They include altered
facts, negation, ambiguous evidence and missing offered answers, but only six
closely related templates. They are neither independently human-reviewed nor a
replacement for the original broader 200--500-case target. A human must check
and correct them before their labels can be treated as reviewed ground truth;
additional coverage is still needed. No generated approval or model-scored packet
is being passed off as human review.

The native prefix, diagnostic and resource checks are complete for this bounded
scope. **Issue #2 remains open on the human-review/evaluation requirement**, and
neither tested scorer is promoted.

[update]: https://github.com/rcarmo/go-pherence/issues/2#issuecomment-5742293093
[results]: prefix-results.json
[mass]: label-mass-policy.md
