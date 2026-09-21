## Exact-prefix validation policy

This development check uses only the installed pinned Qwen3-4B instruction
checkpoint and existing synthetic direct-reference questions. Final-test and
training data remain closed. It adds no learning recipe or model assets.

For each fixture, independently render/tokenise its complete prompt, retain the
longest exact common token prefix of its sibling group, then execute each suffix
alone, serially from retained K/V and in a packed suffix batch. The shared state
contains only identical instruction/evidence bytes before the first differing
question; sibling text is never concatenated into that state. Assert this at the
prompt-token boundary, not by assuming string concatenation preserves tokens.

The native path remains BF16 resident with compensated F32 matrix arithmetic and
F32 K/V. Gates fixed before new execution: maximum selected-logit difference
0.005, RMS 0.0002, maximum conditional-probability movement 0.001 against
independent native execution. Existing independent Transformers gates stay
unchanged. Report every winner change; any change outside a reference logit gap
of 0.01 fails. Near ties are reported separately, not used to loosen logit gates.
Include a constructed exact/near-tie arithmetic fixture if real model probes do
not supply one; do not describe synthetic arithmetic as observed model behaviour.

Test one question, multiple unequal-length questions, reversed question order,
irrelevant and contradictory-instruction siblings, and re-execution after each.
Packed execution uses ragged offsets without padded rows: last-real-token indices
are explicit. A padded-input admission regression must reject padding disguised
as real suffix tokens/lengths; no logits may be read from padding. Record retained
prefix checksums before and after normal, cancelled and failed work.

Prefix K/V is owned by its encoder, bounded to one live prefix and released before
replacement or encoder close. Requests admit at most eight branches, 512 total
suffix tokens, 2,048 effective prefix-plus-suffix tokens and 128 candidate tokens.
The in-flight plus queued bound is 32 branches, 2,048 suffix tokens, 4,096 effective
tokens and 512 candidates. Admission is nonblocking rejection on budget excess;
GPU access is safely serialised. Cancellation is cooperative between layers,
with CUDA work drained before returning; it is not kernel preemption. Rejected
or cancelled calls return no partial result, and must leave the prefix reusable.

Measure independent, KV-cached serial and packed suffix paths on identical single
and multi-question workloads, five warm repetitions. Report prefix construction,
tokenisation, suffix/prefill, selected projection, transfer bytes, request latency,
questions/sec and GPU allocation/free-memory observations. Startup hash/load and
prefix-cache creation are separate costs. Packing projections is not concurrent
GPU request execution. No full-vocabulary projection or continuation occurs.

[Experiment status](README.md) | [Additional acceptance](https://github.com/rcarmo/go-pherence/issues/2#issuecomment-5742293093)
