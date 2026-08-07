# A Practical LLM Inference Blueprint

Fast inference work has an unfortunate tendency to begin with the most impressive subsystem in the room. Continuous batching, paged KV, CUDA Graphs and speculative decoding all look like sensible starting points--until they are attached to a generator that cannot pause after one token, restore its cache exactly, or tell the scheduler who owns its buffers.

This blueprint starts lower down. It is meant for a local or single-node inference runtime that already produces correct output and needs to improve latency, throughput and concurrency without turning the original implementation into an archaeological layer.

The ordering matters. Each phase establishes the contract needed by the next one, and every optimisation has a measured promotion gate. A rejected experiment is a result; carrying it into serving because another engine benefits from it is merely optimism with extra allocations.

## The invariant

Keep one trusted path as the oracle. It may be scalar, slow and boring, but it must remain callable after faster paths land. For a fixed model, tokenizer, prompt, sampler configuration and random seed, candidate implementations should preserve:

* prompt wrapping and tokenisation;
* token-by-token logits within the format's stated tolerance;
* greedy tokens exactly;
* stop-token, length and cancellation behaviour;
* KV state after prefill, decode, checkpoint and restore;
* independent request state under concurrent use.

Do not make `Generate` stateful in place if callers already depend on it. Extract a request-scoped session beside it, prove parity, then move serving onto the session. The old path is useful precisely because it did not move.

## Freeze the workload before touching the engine

A benchmark without pinned assets is a weather report. Record the model revision and digest, tokenizer, prompt bytes, chat template, output limit, sampler settings, backend, thread count, CPU feature set, GPU model and driver. Keep short, long and mixed prompts; repeated prefixes; early-stop cases; malformed state; and at least one context that crosses every sliding/full-attention boundary in the model.

Measure the user-visible path rather than a convenient kernel alone:

| Measure | What it catches |
|---|---|
| TTFT | tokenisation, queueing, prefix restore and prefill interference |
| ITL p50/p95/p99 | decode stalls, scheduler fairness and batch tail effects |
| TPOT and E2E | aggregate cost and regressions hidden by one latency number |
| Aggregate tokens/s | useful batching and projection amortisation |
| Queue time and SLO goodput | overload behaviour rather than peak throughput |
| Allocations and RSS/VRAM | scratch churn, cache growth and hidden copies |
| KV used/reserved bytes | capacity slack and the case--or lack of one--for paging |
| Cancellation latency | whether work is genuinely interruptible |
| Cycles, instructions and cache misses | CPU changes that elapsed time alone cannot explain |
| Kernel launches and transfers | GPU paths that spend their gain crossing the host boundary |

Use fixed arrivals for reproducibility and Poisson or Gamma arrivals for queueing behaviour. Run concurrency 1/2/4/8 before inventing a larger matrix. Report the complete workload and sample count beside every number; a one-iteration real-model gate is useful, but it is not a stable throughput claim.

## Make inference resumable

The minimum useful request contract is small:

```go
type Session interface {
    PrefillChunk(tokens []int) (PrefillResult, error)
    DecodeStep() (DecodeResult, error)
    Checkpoint() (Checkpoint, error)
    Restore(Checkpoint) error
    Finished() (bool, FinishReason)
    Close() error
}
```

A practical implementation also needs bounded prefill (`PrefillNext(limit)`), output and position accounting, stop/sampling state, and explicit backend identity. Checkpoints belong to one session and must be rejected by every other session.

Ownership rules are more important than method names. A session owns its KV, sampler state, pending logits, scratch and cancellation lifecycle. Immutable weights may be shared. Mutable GPU buffers may not be shared unless the backend has a request-indexed allocation scheme and proves that two sessions cannot overwrite one another.

Prefill and decode should produce the same boundary logits as the oracle. Checkpoint/restore tests need to repeat the cycle, not merely restore once--capacity growth, ring cursors and pending-token state tend to fail on the second pass.

## Put serving on sessions

Once decoding can yield after one token, the HTTP layer can stop pretending that streaming means flushing a completed response in small pieces. Create one session per request, narrow model-selection locks to selection rather than execution, emit each token immediately, check cancellation between bounded units of work, and call `Close` on every exit path.

Keep shared execution serialised until backend ownership is proven. Request-scoped state does not automatically make model code re-entrant; global scratch, mutable weight caches and one GPU KV arena are common counterexamples.

The serving benchmark should run through the real HTTP/SSE path. Direct model benchmarks are excellent diagnostics, but they cannot measure queue delay, disconnect handling or lock contention.

## Cache prompt state, not just tokens

Prefix reuse is the first optimisation that can turn seconds into milliseconds without changing transformer arithmetic. The cache key must include more than the token prefix:

```text
model + checkpoint + config + backend + weight layout +
KV policy/precision + RoPE policy + cache salt + token blocks
```

Hash collisions need token-block verification. Entries need byte accounting, eviction and immutable snapshot ownership. Restore the longest complete block prefix and prefill only the suffix; partial blocks are recomputed.

A snapshot includes every piece of state required to reproduce boundary logits: mixed sliding/full KV, shared-KV ownership, ring cursors, per-layer metadata and any model-specific side state. Exact output, restore latency, bytes copied or shared, eviction under a tiny budget and concurrent references are the promotion gates.

## Schedule bounded work

A small scheduler is enough to expose the real trade-offs. Start with FCFS waiting and running queues, decode-first selection, cancellation, a maximum active-session count, and separate decode/prefill token budgets. Leave preemption out.

Chunked prefill only helps if a chunk is short enough to let decoders run. Sweep the fairness quantum against a monolithic control and measure short-request TTFT, active-decoder ITL and total wall time. A policy that improves TTFT while making p99 ITL intolerable has changed which request loses; it has not solved interference.

Retain the smallest policy that moves the target percentile without unacceptable throughput loss. Workload-specific controls are preferable to a universal default backed by one prompt pair.

## Batch only what is independently owned

Static batching is a useful experiment before continuous batching. Represent batch inputs explicitly--flattened tokens, positions and output lengths work well for fixed sizes 1/2/4/8--and compare every row against an independent-session oracle.

Start at safe shared lowerings such as final normalisation and the LM head. Move into projections, attention and transformer layers only when each request has independent KV and scratch ownership. Requests finishing at different times must not leave stale rows contributing logits or cache writes.

CPU dispatch should be shape-aware. M=1 decode usually favours GEMV; M=2/4/8 may favour quantise-once row-parallel projection or GEMM. Dispatch on quantisation, input width, output width and batch size, and preserve alignment, tails and reduction order. Keep unsupported or regressive shapes on the established fallback.

For each batch size, report aggregate tokens/s and per-request ITL. A batch that improves total time by 7% while turning one-token latency into 20 seconds belongs in a benchmark, not in serving.

## Treat GPU Graphs as a residency problem

Capture/replay pays off when shapes and addresses are stable and the captured region stays on device. Begin with one fixed batch-1 decode segment. Measure eager latency, capture cost, warm replay, shape changes, parity and teardown.

Full-model capture is premature while the path contains host decisions, transient allocations, CPU fallbacks, KV shadow copies or logits downloads. Remove those boundaries because their measured cost matters; do not build a graph-shaped wrapper around them.

Stateful GPU batching has the same ownership gate as CPU batching, only less forgiving. One mutable device KV set means one active request, regardless of how attractive the matrix dimensions look.

## Add speculation through the session boundary

Speculative decoding should look like a variable-token session operation, not a second generation engine. The verifier stages `input + drafted tokens` into ordinary request KV, computes one logits row per verifier position, applies greedy acceptance plus the bonus token, then commits exactly the accepted KV prefix. Any failure restores the pre-verification checkpoint.

A suitable extension is conceptually:

```go
type VerifyingSession interface {
    Session
    Verify(drafted []int, sampler Sampler) (VerifyResult, error)
}
```

`VerifyResult` needs drafted, accepted and bonus accounting, emitted tokens, logits or residual-sampling inputs, and a finish reason. Greedy parity comes first. Non-greedy residual sampling should land only after the acceptance and KV commit machinery is shared with ordinary decoding.

Do not maintain separate speculative KV types unless the backend genuinely requires them. Checkpoint, stage, commit-prefix and restore are ordinary cache operations--using two implementations is an invitation to cursor drift.

## Measure KV before paging it

Expose logical and reserved KV bytes per session and per owning layer. Shared-KV consumer layers should report zero; source layers own the storage. Track capacity slack across prompt lengths and while 1/2/4/8 sessions are resident at once.

Linear request-owned slices are often good enough. If reserved bytes scale linearly, proportional slack falls with context length and allocation churn stops after growth, a block pool adds page tables, reference counting and another attention path without fixing a measured problem.

Paged KV earns a prototype when reserved-versus-used waste or allocation churn materially limits concurrency. If that gate trips, implement logical CPU pages first, retain linear/ring KV as the oracle, add reference-counted prefix blocks, and only then attempt device page tables and paged attention.

## Promotion rules

An optimisation is retained when it passes correctness and moves the metric it was meant to move on the real workload. The decision record should say what happened plainly:

| Result | Action |
|---|---|
| Exact or tolerance parity fails | reject |
| Batch 1 regresses | keep the established M=1 path |
| Only one shape improves | dispatch only that shape |
| Aggregate throughput rises but ITL misses the SLO | keep experimental, do not serve |
| Memory rises beyond the concurrency budget | reject or reduce residency |
| GPU replay wins only for a fixed segment | retain the segment; do not claim full capture |
| Paging gate does not trip | keep linear/ring KV |
| Hardware is unavailable | record the missing gate; do not manufacture a result |

Every retained path also needs race coverage, malformed-input coverage, feature-disabled fallback tests, architecture cross-builds and a way to identify which dispatch ran. Broad repository suites are useful, but unrelated hardware packages should not obscure focused ownership and parity gates--record both outcomes.

## What to postpone

Recompute preemption, disaggregated prefill/decode, external KV services, tensor/pipeline/data parallel orchestration and distributed control planes solve scale problems. They also multiply failure modes. Keep them out until the local scheduler, batching and KV measurements show a concrete bottleneck that cannot be removed more simply.

The useful pattern is deliberately unglamorous: freeze the workload, extract state, stream bounded work, cache exact prefixes, schedule fairly, batch measured shapes, and only then consider machinery that changes memory addressing or execution topology. This leaves a runtime that can absorb the next optimisation without surrendering the oracle that tells you whether it worked.
