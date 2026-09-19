## Candidate text for independently encoded heads

A late-interaction head encodes each candidate without the question or the other
candidates. Its online contract must therefore distinguish a complete alternative
from a fragment whose meaning is supplied elsewhere. Direct joint-prompt scoring
does not have the same structural limitation.

`feature-score` and `head-online-bench` require an explicit request field:

```json
{
  "candidate_contract": "self-contained-v1",
  "evidence": "Mara put the amber key in the blue drawer. The red drawer is empty.",
  "question": "Where is the amber key?",
  "candidates": [
    {"id": "blue", "text": "The amber key is in the blue drawer."},
    {"id": "red", "text": "The amber key is in the red drawer."},
    {"id": "unknown", "text": "The amber key's location is not specified by the evidence."}
  ],
  "temperature": 1
}
```

The author promises that each candidate names the entity, relation or action
being considered and can be interpreted in isolation. Candidates must not point
to another option, rely on order, use answer codes as text, or require resolving
an omitted subject from the question. Unknown/missing-answer alternatives must
state what is unknown; they are not confidence-based deferral. Stable IDs are
opaque application identifiers, not semantic features or answer positions.

The validator enforces distinct nonblank text/IDs and rejects obvious positional
or context-dependent forms such as `A`, `option B`, `yes`, `no`, `that` and `all
of the above`. It **cannot prove semantic self-containment**. A declaration is
not a human review: an apparently complete sentence can still contain an
unresolved pronoun, ambiguous name or unsupported premise. Reports explicitly
set `semantic_review_verified=false` unless an external review process exists.

For unchanged benchmark candidates, callers may explicitly declare
`benchmark-fragments-v1`. This is an exploratory compatibility mode, not a pass
of the self-contained contract. The completed frozen-head study used the original
benchmark alternatives, many of which are short noun or intent fragments;
rewriting those now would change the task and invalidate the recorded comparison.
Its historical request hashes and trained feature contracts are preserved. Any
proposition rewrite needs its own manifest and evaluation, and possibly retraining.

## Permutations preserve logits, not necessarily a tied winner

For a permutation `p`, candidate-ID-mapped logits should obey
`score(p(options))[j] == score(options)[p[j]]`. The head does not encode option
position, so this property is tested directly rather than inferred from equal
accuracy. Unit tests exhaust all permutations for two, three and four candidates,
including mixed-length padded batches and F32/FP16 features. They also remap the
gold index and check NLL/Brier consistency.

`head-permutation` tests each original validation request with reversal and every
nonzero cyclic rotation, maps by stable ID, and fails on any logit difference or
changed unique argmax. For the completed three-seed study: **288 permutations per
seed, zero maximum mapped-logit difference, zero unique-winner changes, and zero
strict-tie originals**. A strict tie is different: the API selects the first
supplied candidate, so permuting tied candidates can change the returned ID while
preserving every mapped logit. Tests and reports state that policy separately.

## Three online timing scenarios

`head-online-bench` measures the same request and head under three explicit
conditions. It counts encoder calls and requires bit-identical logits across
conditions; reused objects contain unprojected frozen features, not trainable
attention keys or values.

| Scenario | Context forwards | Candidate forwards | Included work |
|---|---:|---:|---|
| Fresh context, fresh candidates | 1 | Number of candidates | Tokenisation, all GPU feature extraction/pooling, CPU head |
| Fresh context, reused candidates | 1 | 0 | Context tokenisation/extraction, candidate feature lookup, CPU head |
| Cached context, cached candidates | 0 | 0 | In-memory feature lookup and CPU head |

Five measured requests per scenario, using the complete-statement example above
(27 context tokens; candidate lengths 9, 9 and 12), gave:

| Scenario | p50 | p95 |
|---|---:|---:|
| Fresh context and candidates | 2.464879 s | 2.480540 s |
| Fresh context, reused candidates | 0.801313 s | 0.801477 s |
| Fully cached in RAM | 0.000690 s | 0.000734 s |

Candidate-feature setup took 1.6622 seconds and context-feature setup 0.7921
seconds; both are reported separately. The model was already resident, the OS
file cache was warm, and network/queue time is excluded. Startup hashing/loading
are recorded separately. Fully cached means RAM, not the disk-cache-plus-head
measurements in the [head report][head]. Five samples do not establish tail
latency under load.

All three scenarios produced identical logits. The head still selected the red
drawer, so complete candidate statements and lower latency did not establish
quality on this probe. It is machine-authored, not independently reviewed.
Feature reuse here is **not transformer-prefix/KV reuse**; those branch-isolation
checks remain a separate pending optimisation experiment.

[Head experiment and reproduction][head] | [Machine-readable observations][data]

[head]: frozen-head-study.md
[data]: frozen-head-results.json
