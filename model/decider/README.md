# Native Decider

`model/decider` implements the released typed scoring contract of [`Mapika/decider-0.8b`](https://huggingface.co/Mapika/decider-0.8b) in native Go. Source revision `c4daaac28af9fea95d627015cffa2dd5a5926ee6`, model revision `1ea54127d3bd52f6d753d9257b32a6380b873907`, and base revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` are pinned. Source, model, and base model are Apache-2.0.

The package runs the existing Qwen3.5 hybrid implementation (18 gated-delta linear-attention and six full-attention layers), reads the hidden state at `Answer: (`, and projects only the supplied one-token labels. `SystemOne` accepts ordered questions and criteria, supports Choice, Noul and Score, and scores each question independently. Score levels are isolated by default, matching the release: each level is judged as a separate yes/no question and the positive fits are normalized.

```go
r, err := decider.Load(modelDir, 32768)
if err != nil { /* handle */ }
defer r.Close()

out, err := r.SystemOne(
    decider.Object{
        {Name: "ticket", Value: "I was charged twice and want a refund."},
        {Name: "priority", Value: 2},
    },
    []decider.NamedQuestion{
        {ID: "team", Question: decider.Question{
            Type: decider.Choice,
            Instructions: "Which team should handle this?",
            Criteria: []decider.Criterion{
                {Name: "billing", Description: "Charges, invoices, refunds"},
                {Name: "technical", Description: "Bugs and outages"},
                {Name: "other"},
            },
        }},
        {ID: "refund", Question: decider.Question{
            Type: decider.Noul,
            Instructions: "Does the customer request a refund?",
        }},
    },
)
```

`Object` deliberately preserves JSON property order. Plain maps are accepted and deterministically encoded, but Go map order cannot express the caller's original JSON ordering. Questions, criteria, answers and wide-label construction use slices so semantic order is explicit.

## Scope and limits

- State-first independent scoring is the correctness path. The upstream schema-first cache, packed dependent questions, CUDA graphs, FP8, vision, training and RL are not implemented.
- The production `Load` path admits the exact released 0.8B architecture and tied embedding/LM head. It rejects incompatible configs, tensor shapes, tokenizer label tables and unsupported release metadata. `LoadFixture` retains the same structural checks while permitting tiny synthetic geometry for tests.
- The released fixture covers exact token IDs and raw Choice/Noul/isolated-Score logits. Native logits are admitted within `0.12` of Transformers BF16. Package statement coverage is 90.0% without released assets.
- This is CPU compatibility, not a throughput claim. Five released rows take about 25 seconds and peak near 5.0 GiB RSS on the validation host. The tiny synthetic single-choice API takes about 16 µs on Intel and 58 µs on native ARM64, with 7.3 KB and 301 allocations on both; profiles attribute the largest allocation sites to tokenizer regex splitting, Qwen full-attention temporaries and answer-map assembly. Allocation optimization remains follow-up work.
