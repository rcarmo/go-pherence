## Issue #2: the runtime works; the decision model is not ready

Update after the frozen final run: the historical development measurements below
are unchanged. The user approved narrower experimental closure and explicitly
deferred broader human review. The pre-test [policy](final-evaluation-policy.md)
and [freeze](final-freeze.json) were committed before held-out evaluation. The
[final report](final-report-20260921.md) now covers **1,440 of 1,440 originals**
across Instruction direct and frozen heads 7/17/27, with every arm admitting
every row.

The first two attempts stopped at original 677 with CUDA error 719 and repeated
Xid79/Xid154. After explicit recovery authorisation, the evaluator retained all
676 immutable records and wrote only the 764 missing rows. The monitored 42m28s
resume reached 81 C, 137.09 W and 7,399 MiB VRAM, with no Xid, device-query
failure or thermal slowdown. Model assets and the 16-file freeze matched after
the run. InvokeAI and ComfyUI stayed stopped.

The native Qwen3 path fits the RTX 3060, agrees with independent numerical
references, and supports isolated K/V-prefix reuse. The frozen-head experiment
also runs end to end, including cached training, exact interruption and resume,
fresh-text inference and immutable final reporting. Those are bounded engineering
results. The heads did not produce a useful decision model.

On the final-test cohort, direct Instruction reached **81.25% accuracy**. Heads
7, 17 and 27 reached **21.11%**, **22.85%** and **22.64%**, for a **22.20%** mean
against the pooled **23.52%** random expectation. Instruction's frozen calibration
reduced pooled NLL from 2.410 to 0.550 and ECE from 0.171 to 0.046 without changing
decisions. The heads' temperatures reached the ceiling near 20; their flatter
probabilities did not repair their accuracy.

The stronger Instruction result does not erase its development control failure:
reversing alternatives changed 8 of 60 answers, above the predeclared 10% limit.
The heads remain below random on average and failed the evidence-dependence and
expansion gates. Neither configuration is promoted. Issue [#2][issue] can close
as the approved negative research result after this report and its checks are
published; future tuning needs a new selection protocol.

This report consolidates the recorded experiment through 2026-09-21. Broader
human-reviewed and transfer/stress coverage was deferred and is not counted as
performed.

## What was actually evaluated

The prepared dataset has 3,839 training, 478 validation, 482 calibration and 1,440
final-test examples, with pinned sources, grouped splitting, duplicate handling,
licence records and checksummed provenance. Preparation is broader than the
executed study: the direct comparison used **60 validation originals**, 12 from
each source/config, and **228 requests** after option-order and evidence controls.
The separate calibration pilot used another **60 originals**. The heads trained
on **160 balanced examples**, not the entire prepared training split.

The tasks were ARC Easy, ARC Challenge, CommonsenseQA, MultiNLI and CLINC. CLINC
means the prepared labelled **8-choice task including OOS**, not full-intent
classification. Each direct checkpoint completed all 228 requests without an
admission rejection. Final-test examples were not used for prompt selection,
training decisions, calibration, prefix work or threshold selection.

| Method | Normal validation accuracy | Answers changed by reversing options |
|---|---:|---:|
| Uniform random, expected | 23.17% | Not applicable |
| Direct Qwen3-4B-Base | 60.00% | 34 / 60 |
| Direct Qwen3-4B Instruction | 80.00% | 8 / 60 |
| Frozen head, seed 7 | 18.33% | 0 / 60 |
| Frozen head, seed 17 | 18.33% | 0 / 60 |
| Frozen head, seed 27 | 16.67% | 0 / 60 |

The instruction scorer's 13.33% order-change rate exceeds the predeclared 10%
limit. Higher accuracy with reversed options does not erase that failure.
Twelve examples per task also leave substantial sampling uncertainty, and
public benchmark familiarity is a limitation.

The heads used the same instruction backbone and candidate sets, but encoded
context and candidates independently instead of using the direct scorer's joint
chat prompt. All three passed **288 explicit permutations**, with zero logit
difference after mapping back to stable candidate IDs. That establishes
permutation equivariance for these checks, not useful reasoning. CLINC showed no
normal-over-shuffled-evidence improvement in any seed. The fixed five-epoch,
rank-64 recipe failed its expansion gate; it was not retuned until it passed.

## Confidence is conditional on the offered labels

Temperature scaling fitted only on the separate calibration sample produced
**T=6.5667** for the direct instruction scorer. Applied unchanged to validation,
it reduced NLL from **3.1725 to 0.6493** and ECE from **0.1912 to 0.1084**. Accuracy
and option-order sensitivity did not change. The head temperatures reached the
search ceiling of 20; their less-confident guesses did not become better answers.
No confidence-based deferral threshold was approved.

The offline full-vocabulary diagnostic exposed another distinction. One of five
synthetic reference prompts assigned **99.9749% conditional confidence** to its
winning permitted answer, while *all permitted answer tokens together* carried
only **0.001127% of full-vocabulary probability**. Conditional entropy was low,
but the model put nearly all its vocabulary probability elsewhere. This is a
reported diagnostic, not a safety rule or calibrated correctness estimate.

Full-vocabulary normalisation runs only in the offline CPU reference tool. The
native selected-row scoring path does not allocate vocabulary logits, sample,
decode a continuation or parse generated answers. Its probabilities remain
explicitly conditional on complete, finite candidate logits.

## What the implementation now guarantees within the tested scope

The GPU keeps BF16 transformer matrices resident with bounded F32 scratch, rather
than expanding the whole model to F32. Independent instruction-checkpoint
fixtures retain the original tolerances: maximum hidden-state error **0.00447845**
against a 0.005 limit, and maximum selected-logit error **6.49e-5**. Exact prompt
text, tokenizer IDs and the supported no-thinking template branch are checked.
The older CPU long-input path remains unpromoted under this fixture.

The feature cache stores unprojected context rows and deduplicated F32-pooled
options, bound to model-file identity, tokenisation, representation, dtype and
input metadata. Corruption, stale identities and offline misses fail explicitly.
The real interrupted seed-7 run produced **byte-identical optimiser state and
checkpoint files** to an uninterrupted run. Fresh GPU extraction reproduced the
cached head logits exactly. F32 and FP16 caches together used **160.2 MB**; the
FP16 ablation changed no decisions, but was not adopted for training.

True prefix reuse retains immutable per-layer K/V, not candidate features. Serial
and packed suffix execution matched independent logits and probabilities exactly
on six synthetic probes, with both 26-token template and 38-token shared-evidence
prefix checks. Tests include contradictory and irrelevant siblings, question
ordering, unequal lengths, last-real-token selection, mixed concurrent callers,
cancellation, rejection recovery and interrupted prefix construction. Retained
prefix checksums and allocation cleanup passed. Real model probes were not near
ties; exact/one-ULP tie coverage is an explicitly synthetic arithmetic test.

The prefix API bounds aggregate tokens, candidates and branches, including
queued work. GPU access is serialised, and cancellation is cooperative between
layers with submitted work drained before returning. This is neither kernel
preemption nor a claim that concurrent API calls execute in parallel on the GPU.

## Three different latency questions

Fresh frozen-head inference still needs expensive backbone work. On one
27-token context with three complete-statement candidates, five repetitions gave:

| Head scenario | p50 | What is reused |
|---|---:|---|
| Fresh context and candidates | 2.465 s | Resident model only |
| Fresh context, cached candidate features | 0.801 s | Candidate representations |
| Context and candidates cached in RAM | 0.690 ms | All frozen features |

All three returned identical logits -- and selected the wrong drawer on the
key-location probe. Sub-millisecond cached head execution is not a fresh-input
latency claim, and rewriting candidates as complete statements did not fix this
head's answer on that probe. The self-contained-candidate declaration rejects
obvious positional/dependent forms but cannot prove semantic adequacy or human
review. Historical benchmark fragments retain their explicit compatibility label.

The separate direct-scoring prefix benchmark measures **actual transformer K/V
reuse** on identical workloads:

| Questions | Fresh independent p50 | K/V-cached serial p50 | K/V-cached packed p50 |
|---:|---:|---:|---:|
| 1 | 1.674 s | 1.195 s | 1.180 s |
| 6 | 11.118 s | 8.235 s | 6.334 s |

Six-question times are group latency, not latency per question. The packed path
reduces that group's warm p50 by about **43%**, excluding **0.847 s** to construct
the prefix. Model hashing/loading, cache setup, transfers and component timings
are reported separately. These are five-repeat, warm-resident measurements, not
production tail-latency estimates.

Base encoder-owned allocation is **7.465 GB**; the measured prefix and its
workspace add **11.862 MB**. The run left about **4.49 GB** GPU memory free and
restored the pre-load free-memory count after close. Only one full checkpoint is
retained: Base's reproducible weight shards were removed after preserving its
results and references. Model assets remain below 12 GiB and caches below 12 GiB.

## Acceptance is not a single checkbox

| Area | Current disposition |
|---|---|
| Native extraction, selected-row scoring, cache, head training, resume and fresh inference | Implemented and exercised on the approved RTX 3060 |
| Numerical references, isolated prefix execution and bounded request-state tests | Passed for the documented fixtures and limits |
| Direct scorer quality | Failed option-order requirement; not promoted |
| Frozen-head learning and context dependence | Failed the fixed pilot; expansion stopped |
| Calibration and probability diagnostics | Measured on separate development samples; no deferral policy approved |
| Independent human-reviewed fresh cases | Explicitly deferred by the user; 24 proposed cases stay unreviewed/unscored |
| Frozen final evaluation | Completed 1,440/1,440; all arms admitted all rows; Instruction 81.25%, head mean 22.20%, pooled random 23.52% |

The packet has six closely related templates with four entity substitutions;
it does not satisfy the original 200--500-case target or broad transfer coverage.
That coverage was deferred, not waived into a pass. The current operational
blocker is GPU recovery, and eventual completion of the frozen final run cannot
retrospectively make the option-order or head-learning failures pass.

The original ticket contains a wider comparison and stress-test matrix than this
bounded study: withheld domains/templates, systematic paraphrase and
distractor-count tests, larger/full-intent candidate sets, semantic-class
precision/recall and confusion matrices, and a separate relevance reranker where
appropriate. The current reports do not contain those results. The direct scorer
is a joint-prompt baseline, not a completed separate relevance-reranker
comparison. The user explicitly accepted the narrower negative research closure;
ticket closure must retain those items as deferred work rather than completed
coverage.

The earlier verification had a clean worktree, matching local/remote revisions
and passing model-layout CI. Its GPU-serving and disk-space statements were
point-in-time observations, not current health guarantees. The later audit has
90 passing host race packages, 71 without tests and compile-only ARM64/RISC-V
checks. The recovered device completed the missing-row run under a fail-closed
watchdog; that successful run does not identify the earlier Xid79 cause. Another
training sweep is outside the approved scope.

## Supporting reports

The [final report][final], [direct comparison][direct], [frozen-head
experiment][head], [candidate contract][candidate] and [prefix/label-mass
study][prefix] carry the reproduction commands, per-task measurements and links
to machine-readable results. The
[additional acceptance comment][acceptance] defines the native isolation and
probability work. This report's source snapshot is [bd19e97d][source], with its
[passing CI run][ci]; the report itself does not change runtime code.

[issue]: https://github.com/rcarmo/go-pherence/issues/2
[final]: final-report-20260921.md
[direct]: direct-study.md
[head]: frozen-head-study.md
[candidate]: candidate-contract.md
[prefix]: prefix-study.md
[acceptance]: https://github.com/rcarmo/go-pherence/issues/2#issuecomment-5742293093
[source]: https://github.com/rcarmo/go-pherence/tree/bd19e97de273738795c212d894416b8a85082197
[ci]: https://github.com/rcarmo/go-pherence/actions/runs/35447337292
