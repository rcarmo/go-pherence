# Gemma4 baseline and final disposition

Date: 2026-08-07. This is the frozen baseline for the vLLM-leverage programme. It distinguishes measurements produced by the request-scoped Gemma4 session from bounded component experiments and from unavailable backend gates.

## Asset and output fixture

The target is `google/gemma-4-E4B-it-qat-q4_0-gguf` revision `4b4a2c1d584be7264f87aac328a1bc739ce81b6c`. The local GGUF is 5,154,941,280 bytes with SHA-256 `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`.

The native session fixture bypasses tokenizer and chat-template ambiguity deliberately:

| Field | Frozen value |
|---|---|
| Input token | `[10979]` |
| Prepared prompt | `[2, 10979]` |
| Greedy output tokens | `[106, 236789]` |
| Maximum generated tokens | `2` |
| Backend | `simd` |

`TestGemma4DecodeSessionUpdatedRealGGUFTwoSteps` now checks these constants against both legacy generation and the request-scoped session, restores the pre-decode checkpoint, and checks the same two-token trajectory again.

## Measured CPU evidence

Measurements used an Intel i7-12700 with six logical CPUs exposed and AVX2 available. Model load is excluded where the individual report says so.

| Workload | Result | Source |
|---|---|---|
| Frozen two-token legacy/session/restore gate | 10.1--12.0s observed test wall time | real-session gate; includes one legacy trajectory, one session trajectory and one restored replay |
| Prompt-cache cold prefill | 17.8--20.6s | prompt-cache report |
| Full prefix restore | 0.40--0.46ms | prompt-cache report |
| Partial block-2 restore plus suffix prefill | 5.16s | prompt-cache report |
| Scheduler short-request TTFT proxy | 27.28s monolithic; 19.82s quantum 1 | scheduler-interference report |
| Scheduler active ITL proxy | 2.88s monolithic; 5.58s quantum 1 | scheduler-interference report |
| Scheduler total | 34.19s monolithic; 36.13s quantum 1 | scheduler-interference report |
| Static batch B2/B4/B8 aggregate time | 18.7%/12.7%/7.0% better than same-batch sequential tail | static-batch report |
| Static batch per-request ITL | 6.3--21.1s | static-batch report |
| Concurrent request-owned KV reserved | 0.344/0.688/1.376/2.753 MiB for 1/2/4/8 sessions | cumulative report |

The serving harness records TTFT, ITL, TPOT, E2E, throughput, queue time, cancellation and SLO goodput for fixed and Gamma/Poisson arrivals. A second full live matrix was not promoted as baseline data: the session supports only SIMD, and mixing legacy scalar or globally mutable NVIDIA execution into one table would imply backend parity that does not exist.

Allocations are retained with the kernel measurements: B8 Q4 projection reuse reduced the measured path from roughly 128--133 allocations to 24--25. RSS is represented by request-owned KV and prompt-cache byte accounting rather than a process-wide number polluted by the 5.15 GB mapped model. Cancellation has deterministic scheduler/server tests; no wall-clock cancellation claim is made because cancellation is observed only between synchronous model steps.

Hardware performance counters are unavailable on this host: `perf_event_paranoid=4` denies cycles, instructions and cache-miss events. The SIMD backend does not expose stable dispatch counters. These fields are recorded as unavailable rather than replaced with estimates.

## NVIDIA evidence and boundary

The available RTX 3060 has 12,288 MiB and driver 580.173.02. The fixed three-kernel CUDA Graph segment produced exact output, 9.94--10.01 microsecond warm replay, 12.50--13.42 microsecond eager execution and 13.14--14.05 microsecond capture.

This is not a stateful Gemma4 session result. `GPUModel` owns one mutable KV/cache set, while `Gemma4DecodeSession` deliberately rejects the NVIDIA backend. NVIDIA prompt/output, TTFT, ITL, TPOT, E2E, serving throughput and request-scoped VRAM measurements are therefore blocked at the ownership gate. Full graph capture and NVIDIA batching remain rejected until request-owned device KV exists.

## Scalar evidence and boundary

Legacy scalar primitives remain correctness oracles for quantised kernels, but the tailored Gemma4 session accepts only SIMD. Reporting scalar session output or timing would require a separate stateful scalar implementation and would not validate the contract shipped here. Scalar session parity and performance are recorded as unsupported, not pending baseline numbers.

## Final disposition

The retained serving path is request-scoped SIMD with immediate SSE emission, exact prompt snapshots, optional decode-first quantum 1, and conservative serial model execution. B8 Q4/Q6 projection specialisations remain shape-gated. Static serving batches, paged KV, full CUDA Graph capture, NVIDIA stateful batching and distributed mechanisms are not promoted.

The programme's completion criterion is parity-safe measured promotion, not a filled table for backends that fail the state-ownership prerequisite. Unsupported scalar/NVIDIA session rows can become a new programme when they acquire independent mutable state.
