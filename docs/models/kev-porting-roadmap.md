# Kev porting assessment and roadmap

Kev-0.8B is a plausible native inference port because go-pherence already has the required Qwen3.5 hybrid backbone, PEFT LoRA loading and merging, recurrent/full-attention prefix state, and typed decision APIs. The first approved implementation slice should be a pinned 0.8B reference fixture and checkpoint reader. This document records assessment and priorities; it does not approve implementation.

## Reviewed sources

The source review is pinned to [`jaredpalmer/kev@90990a5fac2995b9faa3190f7d437e84f2067768`](https://github.com/jaredpalmer/kev/tree/90990a5fac2995b9faa3190f7d437e84f2067768). Relevant files are:

- [`kev/model.py`](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/model.py): delimiter-safe encoding, question isolation, Qwen3.5 row execution, prefix reuse and pointer readout;
- [`kev/api.py`](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/api.py): TypeSafe-compatible Noul, Choice and Score request/response conversion;
- [`kev/checkpoint.py`](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/checkpoint.py): PEFT adapter loading, pointer-head metadata, temperature and compatibility checks;
- [`kev/serve.py`](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/serve.py): one-request admission, repeated-state cache and diagnostic endpoints;
- [`kev/metrics.py`](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/metrics.py): calibration and selective-risk metrics;
- [`tests/test_model.py`](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/tests/test_model.py): merged/unmerged, full/prefix, packed/row and hybrid-isolation parity checks.

The reviewed model revisions and immutable large-file hashes are:

| Model | Hub revision | Base revision | Adapter SHA-256 | Head SHA-256 |
|---|---|---|---|---|
| `jaredpalmer/kev-0.8b` | `54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | `c81d5716f0af7622d8d2b97013c333d48263ca01113f9a7cf4526e96f6ac0b26` | `39f4343ccccc65e583bbfff0de0e11bfedb849fcfaaf94b50ac4f2b73bc79c65` |
| `jaredpalmer/kev-4b` | `485ace8703592fcf405488b262449990824cfed1` | `1001bb4d826a52d1f399e183466143f4da7b741b` | `9797de69a42188e411b17b7b4fcb66a23374dcebc21d71a7a66f836b5d34df2b` | `d8f796da36ff7bd7c0fb9496b452139bb7851af4fc82b07b500b682d3f721d6a` |
| `jaredpalmer/kev-9b` | `2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c` | `68c46c4b3498877f3ef123c856ecfde50c39f404` | `c02d0d1a70367b5779deb23b28153284b59c067038323cc17ddab5b92dfce72f` | `5528f777437f561cb4a9d56b0278a42b6ae6edd373e7921e52b13adaac988c9f` |

The 0.8B adapter and head are 43,338,624 and 2,103,103 bytes. The 4B artifacts are 129,924,032 and 5,248,767 bytes; the 9B artifacts are 173,178,096 and 8,394,495 bytes. Kev source, adapters, heads and Qwen bases use Apache-2.0. Training datasets retain their own licences and require a separate distribution audit.

## Architecture

Kev encodes a state followed by one or more question branches. Each question contains instructions, delimited options and a final decision token. The pointer head applies separate 256-wide linear projections to the final hidden state at the decision token and each option-end token, then computes scaled dot products. A stored scalar temperature calibrates the resulting softmax without changing its argmax.

Qwen3.5 recurrent DeltaNet layers cannot enforce Kev's block-causal packed mask. Kev therefore computes the state once, clones its full-attention and recurrent cache for each question, and runs each question as an independent causal row. Sibling-question isolation is exact. Released Qwen3.5 checkpoints set `option_isolation=false`; options inside one question can still influence each other and option order can change an answer.

Training uses rank-16 LoRA with alpha 32 and dropout 0.05. The released adapters cover attention, MLP and DeltaNet projections. The base model and adapter are merged for serving where the checkpoint permits it. The vocabulary head is unused.

## Fit with current go-pherence code

This assessment uses go-pherence commit `38fe3fab99781a4f656f7d72396e8cb83796233e`.

| Kev requirement | Existing native surface | Gap |
|---|---|---|
| Qwen3.5 full-attention and DeltaNet execution | `model/qwen/qwen35.go`; released 0.8B use is validated by `model/decider` | No new backbone is needed for Kev-0.8B. Exact Kev logits still need a pinned fixture. |
| PEFT LoRA safetensors | `model/qwen/qwen35_lora.go` and `Qwen35BaseModel.ApplyLoRA` | Kev tensor names start with `base_model.model.layers`; the existing loader expects a different Qwen export prefix. Production loading also needs exact base/revision/target validation. |
| State-prefix reuse | `model/qwen/prompt_cache.go`, `CloneQwen35BaseForwardState`, and byte-budgeted CPU/GPU prompt caches | Kev needs a request-local state snapshot followed by one cloned branch state per question. GPU cache objects currently store state but are not a complete device-resident branch executor. |
| Hidden states for every branch token | `ForwardChunkLayerStreamed` returns one final hidden row per input token | A Kev wrapper must retain only option-end and decision rows and apply the released final-normalisation contract exactly. |
| Pointer head | SIMD dense projection and dot products already exist | No Kev head loader or scorer exists. `head.pt` is a PyTorch pickle archive and must not be deserialised by a production service. |
| Typed questions | `model/decider` already implements Choice, Noul and Score as native ordered types | Kev prompt formatting, direct listwise Score semantics, confidence formula and response shape differ. Reuse types only where it preserves exact Kev behaviour. |
| Calibration and risk reporting | `model/jevlike/calibration.go` already computes NLL, Brier, ECE, reliability and risk/coverage and fits temperature | Kev adds named coverage-at-error and area-under-risk-coverage summaries. Those are small evaluation additions, not runtime prerequisites. |
| TypeSafe HTTP service | No native `/v1/systemone` server exists | Add only after the library API and released-model parity pass. Authentication and public binding remain deployment concerns. |
| Native training | Needle has model-specific LoRA training; Qwen3.5 has inference-side LoRA merge | Full Qwen3.5 adapter and pointer-head training, optimiser state and gradient parity are not implemented. Keep Python training as the reference workflow initially. |

The earlier [Go System One comparison](../validation/go-system-one-kev-20260922.md) remains valid for Go System One: Kev's trained pointer head cannot be attached to the pinned Gemma 4 checkpoint. Repository-wide port feasibility is better because the separate native Qwen3.5 runtime already exists.

## Priorities

### P0 — Freeze the 0.8B contract and independent oracle

This is the next recommended work after explicit approval.

1. Add a source manifest containing the source, model and base revisions above, artifact sizes, SHA-256 values, Apache notices and dataset-licence exclusions.
2. Export `head.pt` with a development-only Python helper using `torch.load(..., weights_only=True)`, then write a safe, bounded native format such as safetensors plus JSON metadata. The 0.8B head has four F32 tensors: `q.weight [256,1024]`, `q.bias [256]`, `k.weight [256,1024]` and `k.bias [256]`.
3. Generate a small independent Transformers fixture with exact request text, token IDs, delimiter positions, option/decision hidden rows, raw pointer logits, calibrated probabilities, prefix state results and typed answers.
4. Include hostile delimiter text, two sibling questions, changed question order, changed option order, repeated-prefix execution and malformed checkpoint metadata.

Acceptance requires exact token IDs and fixed numerical tolerances derived from BF16/F32 reference runs. The fixture must identify Transformers, PyTorch, device, dtype and Kev source commit. This phase does not need an HTTP server or training data download.

### P1 — Port Kev-0.8B inference as `model/kev`

The package should own Kev checkpoint validation, prompt semantics, pointer readout and typed answers while importing the shared Qwen runtime.

- Extend the Qwen LoRA loader with an explicit, tested Kev/PEFT prefix contract. Do not accept arbitrary suffix matching or silently ignored tensors.
- Load the pinned Qwen3.5-0.8B base, merge every declared adapter target, and reject missing, duplicate, undeclared or shape-incompatible tensors transactionally.
- Implement delimiter-safe tokenisation matching Kev's rewrite rule; do not reuse Go System One's reject-only policy if it changes Kev-visible text.
- Prefill the state once. Clone `Qwen35BaseForwardState` for each question and process branch rows independently. Retain only option-end and decision hidden rows.
- Apply the F32 pointer head and stored temperature. Preserve question and option order explicitly.
- Expose a library-level System One call before adding network code.

Required gates are full-versus-prefix parity, repeated prefix reuse, sibling isolation, merged-versus-reference logits, pointer-head parity, cancellation, bounded allocation, close/error rollback, race tests, CPU/SIMD execution and cross-builds. Released-model parity should cover all three question types.

### P1 — Reuse Kev's evaluation discipline

Use the current `model/jevlike` metrics instead of copying parallel implementations. Add coverage at 5% and 1% error, area under the risk-coverage curve and confident-error reporting only if a planned comparison consumes them.

Kev must enter a newly frozen comparison cohort. The completed 1,440-row Jevlike test set and the spent Decider/OpenJEV bake-off cohorts cannot become tuning or model-selection data. The study should compare Kev-0.8B with Decider using stable candidate identities, normal/reversed option order, sibling-question isolation, shuffled/absent evidence, accuracy, NLL, Brier, ECE, risk/coverage, p50/p95 latency and peak RSS. Fit temperature only on a disjoint calibration split; report both raw and released-temperature probabilities.

### P2 — Measure cache and service work after parity

Kev's repeated-state cache is useful when the same state is asked different questions across requests. Decider currently recomputes every state/question row, while go-pherence already has byte-budgeted Qwen CPU and GPU prompt-cache primitives.

Measure cold state prefill, warm prefix reuse, clone cost and retained bytes before adding a service cache. The cache key must include model and adapter identity, dtype, token IDs, position policy and isolation mode. Set an explicit byte budget and NVIDIA headroom; do not copy Kev's four-entry count limit without measuring state size.

A later HTTP package may implement `/v1/systemone`, `/v1/systemone/separate`, `/v1/systemone/permute` and `/v1/models`. Keep it separate from Go System One's `/v1/decision`. Match the TypeSafe wire contract with official-SDK tests before describing it as compatible. Bind locally and retain single admission until memory and concurrency tests justify another policy.

### P2 — Consider Kev-4B after the 0.8B port qualifies

Kev-4B improves published out-of-domain accuracy over Kev-0.8B, but its bf16 serving footprint is reported at about 9 GB before native runtime overhead and cache headroom. It may fit a 12 GB RTX 3060 only with careful admission and measurement. Port it by widening pinned geometry in the same package after 0.8B parity; do not fork a second implementation.

Quantised execution is a separate experiment. Kev publishes no GGUF or quantised adapter merge. Merging in F32/BF16 and then quantising can change pointer logits and calibration, so it needs an independent checkpoint conversion, numerical fixture and quality study.

### P3 — Defer Kev-9B, native training and broad API extensions

Kev reports about 19 GB for 9B bf16 serving, which exceeds the RTX 3060 target before caches and scratch. Defer it until a validated quantised 4B path establishes conversion and calibration rules.

Native Qwen3.5 LoRA and pointer-head training requires backward coverage for attention, DeltaNet, MLP and the pointer head. Existing inference-side LoRA merge does not satisfy that requirement. Continue to use Kev's pinned Python training workflow for producing adapters; import only verified inference artifacts.

Date-fact preprocessing, fine-tuning automation, Modal orchestration and the Next.js playground are application or research tooling. They should not enter the native model port unless a separate use case and test contract require them.

## Limits and non-goals

- Kev's published H100/MI300X latency and Apple M5 measurements use different hardware, models and workloads from go-pherence. They do not predict native RTX 3060 or CPU performance.
- The released checkpoints use `option_isolation=false`. Sibling questions are isolated; options within one question are not permutation invariant.
- Kev serving allows 8,192 tokens although training used at most 384 state tokens and 1,024 tokens for state plus one question. Native admission must preserve this quality caveat.
- A single fitted temperature calibrates an evaluated distribution. It is not a universal accuracy estimate and may not transfer to local tasks.
- Dataset licences vary. Porting inference code and Apache model artifacts does not grant redistribution rights for all training or evaluation data.
- Go System One, Decider and Kev have distinct learned contracts. Shared runtime and evaluation helpers should not collapse their prompt formats, response semantics or validation rules.
