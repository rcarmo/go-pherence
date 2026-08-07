# Gemma4 static DecodeBatch experiment

Date: 2026-08-07. Official E4B QAT Q4_0 GGUF on i7-12700, `GOMAXPROCS=2`, one real decode step per case. Model loading, session creation, one-token prompt prefill, and prefill-boundary decode are outside timing.

The experimental API keeps transformer/KV execution request-local and sequential, then batches the final norm and LM head. This is the only currently safe shared lowering; it is not enabled in serving.

| Batch | Sequential total | Batched-tail total | Batched-tail aggregate | Change vs sequential |
|---:|---:|---:|---:|---:|
| 1 | 2.858s | 3.075s | 0.325 tok/s | -7.6% |
| 2 | 7.475s | 6.299s | 0.318 tok/s | +18.7% |
| 4 | 12.227s | 10.850s | 0.369 tok/s | +12.7% |
| 8 | 22.546s | 21.068s | 0.380 tok/s | +7.0% |

Exact independent-session token and logits oracles pass for batch sizes 1/2/4/8 under the race detector, including divergent prompt lengths and the prefill-boundary fallback. Per-session KV and output state remain independently owned.

Disposition:

- Keep the API explicitly experimental and non-default.
- Reject batch 1: batching overhead regresses latency.
- Do not promote static batching to serving. Batch 4/8 improve aggregate throughput only about 5–9% over the sequential batch-1 reference, while per-request ITL grows to the entire 10.9–21.1s static step because the transformer body remains sequential.
- The batch-tail optimization is useful evidence and an oracle for future AVX2/NVIDIA full-layer batching, but does not meet the latency gate by itself.
