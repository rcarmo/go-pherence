# OpenJEV native validation — 2026-09-21

This milestone implements the released text NLI path of `AlexWortega/openjev` in native Go. The Hugging Face repository declares MIT; the checkpoint declares `Qwen/Qwen3.5-4B` as the family base, while the selected smallest subfolder is the documented Qwen3.5 0.8B full-fine-tune.

## Provenance and contract

- Repository/model revision: `4395b29714015162db6112de91c35688e6e42717`.
- Supported subfolder: `qwen3.5-0.8b-nli-v2s-long`.
- Safetensor SHA-256: `cf6d62a341c0c804f9a926eec71aefc9859adb28978736e757b49bce35d9b8f8` (1,706,036,760 bytes).
- Tokenizer SHA-256: `d73c2c5f7aa0ed522c8d96ef3524739eb61e3c78e74839a2ce4a1c56ea340a20`.
- Architecture: 24-layer Qwen3.5 hybrid text backbone, hidden size 1,024, plus BF16 `score.weight [3,1024]`.
- Input: `Premise: {premise}\nHypothesis: {hypothesis}`, stripped fields, right-truncated to 4,096 tokens.
- Labels: contradiction, entailment, neutral; ordinary softmax with no fitted temperature.

## Numerical evidence

A CPU-only Transformers 5.17.0 oracle records exact text, token IDs, raw logits and probabilities for four rows: clear entailment, clear contradiction, an underdetermined statement, and the released multiple-choice reranking wrapper. Reference outcomes are:

| Pair | Prediction | Probability |
|---|---|---|
| guitar → making music | entailment | 0.9773 |
| blue sky → green sky | contradiction | 0.9807 |
| reads a book → outdoors | contradiction | 0.7124 (neutral 0.2736) |
| photosynthesis → carbon dioxide | entailment | 0.9878 |

Native Go reproduces every token ID and admits every raw logit within `0.12`. The four-row native run takes about 8.5 seconds on the Intel validation host. The CPU-only Transformers oracle took 6.98 seconds and 2.04 GB peak RSS; measurements are not throughput claims.

## Limits and gates

Native support is text-only and deliberately excludes images, the 4B/35B checkpoints, shared-prefix state branching, latent MLP heads, training and quantization. The production loader validates the exact supported geometry, sequence-classification metadata, label order, tokenizer, final norm and score head.

Combined unit/released package statement coverage is **90.9%**. The tiny synthetic scorer measures roughly 11 µs on Intel and 43 µs on native ARM64 CIX P1, with 4.6 KB and 240 allocations on both. Profiles identify Qwen full-attention temporaries as the largest allocation site.

Final gates passed: 119 race-tested packages plus 54 no-test packages, `go vet ./...`, host build, Linux ARM64 and RISC-V builds, 380-document link/build checks, and native ARM64 package execution. The frozen experiment remains unchanged at 16 manifest entries, two binaries and 676 records.
