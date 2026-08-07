# Gemma4 scheduler interference

Date: 2026-08-07. Official E4B QAT Q4_0 GGUF on i7-12700, `GOMAXPROCS=2`, one bounded iteration per policy. Model load is outside timing.

Workload: one active 3-token decoder, one 7-token long prompt, and one 1-token short prompt. Decode budget is 1 token per scheduler step; prefill quantum varies.

| Prefill quantum | Short TTFT proxy | Active ITL proxy | Total |
|---:|---:|---:|---:|
| 1 | 19.82s | 5.58s | 36.13s |
| 2 | 22.19s | 8.26s | 38.23s |
| 4 | 32.22s | 12.83s | 38.82s |
| 64 (monolithic control) | 27.28s | 2.88s | 34.19s |

Disposition:

- Retain quantum 1 as a latency-oriented control: short-request TTFT improves about 27% versus the monolithic control, for about 5.7% more total wall time.
- Do not claim universal ITL improvement. The active-ITL proxy is non-monotonic because this first scheduler executes model work synchronously: a prefill chunk still blocks the next decode step.
- Quanta 2 and 4 are rejected for this workload.
- Revisit ITL after static decode batching or parallel executor work can overlap/amortize request work.

The deterministic scheduler test independently verifies that decode-first quantum-1 scheduling reduces active first-token delay, active ITL and short-request TTFT under controlled per-token costs without exceeding 2x total time.
