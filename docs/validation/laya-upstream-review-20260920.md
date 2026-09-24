# Laya upstream review -- 2026-09-20

The authoritative source is [`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya) at commit `42626c348753fbb17572a813127df2278a1ec527` (2026-09-20). It is Apache-2.0 licensed. The package version is 0.3.4, matching simple-jev's optional dependency pin. The public model repository is `convaiinnovations/laya` at revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, also tagged Apache-2.0 and neither private nor gated.

This is an implementation plan, not a port or qualification claim. No model weights were downloaded and no GPU was used during the review.

## Model contract

Laya is a calibrated “System 1” decision model, not a next-token label scorer. It combines:

* a Hugging Face encoder (`answerdotai/ModernBERT-large` for the English and Typed Decisions checkpoints);
* optional two-layer `nn.TransformerEncoder` decision head with pre-norm, width `d`, `d/64` heads and 4d feed-forward width;
* a three-entry question-type embedding (`choice`, `score`, `noul`) added to every encoder token;
* marker-position gathering for each candidate/rubric option;
* option scorer `LayerNorm(d) -> Linear(d,d) -> GELU -> Linear(d,1)`;
* action/escalation head `Linear(d+4,256) -> GELU -> Linear(256,n_act)` using CLS plus top probability, margin, normalized entropy and option count;
* one temperature per question type.

The sequence format is fixed:

```text
[CLS] <type> question: <instructions> [SEP]
[MASK] option0 [MASK] option1 ... [SEP]
<state> [SEP]
```

Option order is semantic. Choice descriptions preserve `0` and `false`; structured state/criteria use JSON. Score options are rendered as zero-based levels. Noul always has `[false,true]`, with defaults if descriptions are absent. Instructions and options remove literal mask-token text. Option fragments cap at 48 tokens; `head_max_len` budgets the question/options area. State truncates right by default, optionally left. Marker positions beyond `max_len` are removed.

The model softmaxes masked option logits after type-specific temperature scaling. Public answers include the selected option/index, probabilities, normalized-entropy confidence and action/escalation information. Score additionally returns expected level. Noul returns the true probability. This is distinct from simple-jev's fixed nine-bin Noul mapping and next-token logits, and from go-pherence Jevlike's trainable context/option scorer.

Training utilities implement proper log+spherical scoring with ranked-probability penalty for ordered scores, TD(lambda) trajectory targets, temperature buckets and ECE reporting. A future Go inference port need not implement training first, but must not label raw probabilities calibrated until checkpoint temperature and reference outputs match.

## Checkpoints

The Hugging Face revision contains root, `multilingual/` and `typed-decisions/` variants. Approximate safetensor sizes from repository headers are:

| Variant | Safetensor bytes | Encoder |
|---|---:|---|
| English root | 842,609,210 | `answerdotai/ModernBERT-large` |
| Typed Decisions | 842,609,220 | `answerdotai/ModernBERT-large` |
| Multilingual | 643,835,514 | multilingual encoder from its config |

Root config uses `max_len=512`, `head_max_len=192`, two head layers and two actions. Typed Decisions uses `max_len=1024`, `head_max_len=256`, two head layers and gradient checkpointing. Both require `rl_agent_config.json` and `model.safetensors`; tokenizer/encoder assets can be bundled in subdirectories or resolved by config. Loading is strict after compatibility checks.

## Why existing Go BERT is insufficient

`model/bert` implements classic BERT/GTE-small: learned position embeddings, fused QKV, post-attention/output LayerNorm and GELU FFN. ModernBERT uses a different checkpoint namespace and block contract, including rotary positions, alternating local/global attention and ModernBERT-specific normalization/MLP behavior. Treating its tensors as classic BERT would be an architectural substitution, not a loader extension.

A reusable `model/modernbert` encoder is therefore the prerequisite. It should serve Laya and other ModernBERT checkpoints, with model-independent kernels in `backends/simd/runtime` and safetensor/tokenizer I/O in loaders. The Laya typed decision head then belongs in a separate `model/laya` package.

## Proposed implementation sequence and gates

1. Pin and preserve upstream/model licence and revisions. Download one bounded model variant only after storage/memory admission; verify file hashes/revision before use.
2. Inventory ModernBERT config and all safetensor names/shapes. Implement strict bounded loading with no guessed defaults, immutable ownership and checked workspace arithmetic.
3. Implement ModernBERT embeddings, RoPE, local/global bidirectional attention, normalization and MLP against deterministic Python fixtures. Test logits/hidden states layer by layer on FP32 CPU before adding lower precision or SIMD.
4. Reuse backend GEMM/vector kernels and add scalar/SIMD differential coverage for every new primitive/tail. Follow repository allocation and profiling gates; do not copy model logic into backends.
5. Implement sequence construction exactly, including mask-marker positions, option/state budgets, left/right truncation and structured JSON rendering. Add upstream-generated fixtures for empty strings, `0`, `false`, structured criteria, 2/50 options and oversized heads.
6. Implement type embeddings, optional transformer head, marker gather, option scorer, temperature scaling, probability/confidence features and action head. Compare logits, probabilities, confidence, expected score and action output against the pinned Python package.
7. Add strict checkpoint/save metadata and CLI/API inference. Keep Laya, simple-jev and Jevlike endpoints/types separate; they have incompatible scoring semantics.
8. Validate English root first, then Typed Decisions. Multilingual requires its own encoder/tokenizer parity and must not inherit English validation.
9. Run allocation profiles, affected coverage (target at least 90%, 95% for loaders/admission), scalar/ISA fallback gates, native Intel/ARM execution, whole-tree race/vet/build/docs and cross-build checks. Distinguish cross-compilation, numerical parity, released-model inference and benchmark quality.

## Current status

Source and model licensing are compatible, but no native Go ModernBERT implementation or local Laya checkpoint exists in go-pherence. Laya is therefore **scoped and unblocked legally, but not implemented**. The next engineering milestone is ModernBERT tensor/config inventory plus a small upstream hidden-state fixture. simple-jev can call Laya as a backend; its repository gained an Apache-2.0 root licence in `b02aa81c` after this review. An MIT-only reimplementation still needs an independent design and fresh fixtures.

No frozen evaluation artifact, service or GPU state was changed.
