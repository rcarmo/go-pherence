# Native OpenJEV

`model/openjev` implements the text inference contract of [`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev) in native Go. The pinned repository revision is `4395b29714015162db6112de91c35688e6e42717`; the supported checkpoint is `qwen3.5-0.8b-nli-v2s-long` from that revision. The repository is MIT licensed and declares the Qwen3.5 base model.

The package runs the existing native Qwen3.5 hybrid backbone and applies the checkpoint's three-row sequence-classification head to the final token. Label order is fixed by the released config: contradiction, entailment, neutral.

```go
r, err := openjev.Load(modelDir, 4096)
if err != nil { /* handle */ }
defer r.Close()

prediction, err := r.Score(
    "A man is playing a guitar.",
    "Someone is making music.",
)
ranked, err := r.Rerank("Which gas do plants absorb?", []string{"oxygen", "carbon dioxide", "nitrogen"})
```

`Predict` preserves pair order. `Rerank` uses upstream's exact `The correct answer is: {option}` hypotheses and picks maximum entailment probability. `Grade` reproduces the released reference-answer wrapper. Inputs are trimmed and rendered as `Premise: …\nHypothesis: …`; overlong inputs are right-truncated to the configured 4,096-token boundary, matching upstream.

## Scope and limits

- Native support is text-only and targets the smallest released full-fine-tune. Vision, the 4B and 35B variants, shared-prefix batching, latent MLP heads, training and quantization are not claimed.
- The production loader validates sequence-classification metadata, label order, exact 0.8B geometry, tokenizer and score-head shape. `LoadFixture` permits tiny synthetic geometry for bounded tests.
- The released fixture covers exact token IDs and four NLI/reranking rows. Native candidate logits are admitted within `0.12` of Transformers BF16.
- The four-row native parity run takes about nine seconds on the Intel validation host. The tiny synthetic scorer takes about 11 µs on Intel and 43 µs on native ARM64, with 4.6 KB and 240 allocations on both; this is compatibility evidence, not a throughput claim.
- In the [native JEV-equivalent bake-off](../../docs/experiments/jev-port-bakeoff-report-20260921.md), OpenJEV scored 62.96% on 378 untouched prepared validation originals versus Decider's 75.40%. OpenJEV changed 0/378 decisions under reversed candidate order, making exact order invariance its main advantage in that comparison.
