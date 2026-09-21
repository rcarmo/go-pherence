# Native Laya decision model

`model/laya` implements the released [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) typed decision model in native Go. The implementation is pinned to upstream source commit `42626c348753fbb17572a813127df2278a1ec527` and model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`; both declare Apache-2.0.

The package composes the native [`model/modernbert`](../modernbert/README.md) encoder with Laya's question-type embeddings, two pre-norm transformer head layers, marker scorer, calibrated option probabilities/confidence and action head. `BuildSequence` reproduces upstream choice, score and Noul prompts, including per-option/head/state truncation. Public choice criteria are an ordered `[]Criterion`, deliberately not a Go map.

```go
model, cfg, err := laya.Load("checkpoints/laya", "checkpoints/laya/encoder/config.json")
tok, err := modernbert.LoadTokenizer("checkpoints/laya/tokenizer")
response, err := model.SystemOne(tok, state, []laya.NamedQuestion{
    {ID: "colour", Question: laya.Question{
        Type: laya.Choice,
        Instructions: "What colour is the bicycle?",
        Criteria: []laya.Criterion{{ID: "red"}, {ID: "blue"}},
    }},
}, cfg)
```

`SystemOne` currently evaluates questions one by one while preserving request order and reports the sum of unpadded input tokens. The lower-level `Session.ForwardInto` is the performance path: sessions own their workspace, are bounded by sequence/option capacities, and are not safe for concurrent use. Warm tiny and released calls allocate zero bytes.

## Validation

- The deterministic tiny fixture matches pinned upstream option and action logits.
- The released 842,609,210-byte checkpoint matches upstream raw option/action logits for choice, score and Noul questions.
- Exact unpadded token IDs and marker positions match upstream for all three question types; batch padding is represented separately in the oracle.
- Public response assembly is tested against upstream answers: choice `red`, score `1.5291`, Noul `0.7859`, rounded probabilities/confidence and action probability.
- The released checkpoint SHA-256 is `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
- Opt-in released tests bring package statement coverage to 94.0%.
- The [native JEV-equivalent bake-off](../../docs/experiments/jev-port-bakeoff-report-20260921.md) screened Laya at 45.00% on 40 untouched prepared validation originals. Its 2.24 s p50 was the fastest admitted runtime, but lower quality and 3/5 reverse-order changes kept it out of the finalist cohort.

See the [validation record](../../docs/validation/laya-native-20260920.md) for profiling, native architecture results and explicit limitations.
