# Gemma4 prompt-cache results

Date: 2026-08-07

Target: official Gemma4 E4B QAT Q4_0 GGUF, SHA-256 `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`, i7-12700 with `GOMAXPROCS=2`.

Command:

```bash
GO_PHERENCE_GEMMA4_PROMPT_CACHE_REAL=1 \
GO_PHERENCE_GEMMA4_MAIN=models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf \
GOMAXPROCS=2 go test ./model -run '^$' \
  -bench '^BenchmarkGemma4PromptCacheColdFullHitPartialHitRealE4BGGUF$' \
  -benchtime=1x -count=1
```

This is a bounded one-iteration real-model gate, not a stable multi-run throughput claim. Model loading is outside the timed benchmark.

| Block | Case | Prefill | Allocated | Cache bytes | Disposition |
|---:|---|---:|---:|---:|---|
| 2 | cold | 20.63s | 20.0MiB | 3.20MiB | oracle |
| 2 | full hit | 0.462ms | 4.12MiB | 3.20MiB | retain |
| 2 | partial divergent suffix | 5.16s | 10.2MiB | 5.07MiB | retain |
| 4 | cold | 17.78s | 18.2MiB | 2.32MiB | oracle |
| 4 | full hit | 0.400ms | 4.12MiB | 2.32MiB | retain |
| 4 | partial divergent suffix | 10.10s | 12.6MiB | 4.19MiB | retain |

One-token decode after prefill remained approximately 0.08--0.14ms in all cases. Synthetic cold/full/partial tests verify exact output, tiny-budget eviction, longest-prefix behavior, defensive cloning and concurrent references under the race detector.

Block size 2 is the initial Gemma4 recommendation for shared-prefix workloads because it retained a longer useful prefix in the divergent-suffix fixture. Block size remains configurable: smaller blocks consume more entries and metadata, and production selection must be workload-driven.
