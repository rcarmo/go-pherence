# Native Nimble typed scoring

`model/nimble` implements the inference contract of [`bespokelabs/Bespoke-Nimble-9B`](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) in native Go. The published Apache-2.0 adapter is pinned at revision `594dfdcfb6f94e3d0c0db7535180d3c71689169a`; its Qwen3.5-9B base is pinned at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The source checkout used to establish the prompt contract is `f136b3f75721fda4ea961f73993cc50b08488835`.

The package provides:

- ordered flat enum/boolean schema validation;
- exact no-tools, thinking-disabled Qwen chat prompts;
- strict one-token answer-code admission (`A` through `Z`);
- independent per-field native scoring through the Qwen3.5 hybrid recurrent/full-attention runtime;
- candidate-only LM-head projection and typed probabilities at an explicit temperature;
- direct PEFT LoRA loading/merging helpers for Qwen3.5 dense weights, plus the upstream-prescribed merged-checkpoint runtime.

```go
runtime, err := nimble.LoadMerged("/models/nimble-merged", 2048)
defer runtime.Close()
result, err := runtime.Score(context, []nimble.Field{
    {
        Name: "priority", Type: "enum",
        Description: "Urgency based on current business impact.",
        Choices: []any{"HIGH", "LOW"},
    },
}, 1.0)
```

`LoadMerged` expects the pinned adapter to have been merged into the pinned BF16 base with PEFT. This avoids retaining a second 173 MB adapter representation at inference time. The loader accepts standard Hugging Face `[out,in]` linear tensors as well as the existing converted MLX logical layout.

## Limits

The current implementation is CPU-oriented and memory-heavy: the released merged checkpoint occupies about 18 GB on disk and native loading peaks around 59 GB RSS because dense BF16 tensors are converted to FP32 model storage. Released 233-token scoring takes about 3m54s per field on the validation host. This is compatibility evidence, not a throughput claim. The tiny synthetic runtime is about 2.3–2.6 µs on Intel and 9.3 µs on native ARM64, but allocates 1.1 KB / 63 objects per two-token call; zero-allocation inference remains future optimization. Probabilities are option-normalized with temperature 1.0 and are not calibrated correctness estimates. Training, adapter optimization and the upstream 324-example quality evaluation are not re-run here.

Statement coverage is 90.5% without released assets. The tiny synthetic loader exercises the same model-directory, tokenizer, embedding, norm, LM-head and hybrid-layer admission path used by the released checkpoint. Zero-allocation execution remains future optimization.

See the [validation record](../../docs/validation/nimble-native-20260921.md).
