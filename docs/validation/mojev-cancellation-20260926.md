# MoJev accelerated cancellation

SIMD and NVIDIA scorers now expose `ScoreEncodedContext` and `ScoreTextContext`.
Existing methods call the same implementation with `context.Background()`.
Canceled requests return nil output and a context error; cancellation does not
publish partial scores. Nil contexts are rejected.

## Boundaries and resource ownership

A zero-value context-aware mutex permits cancellation while waiting for scorer
scratch. It uses a lazy channel semaphore, no polling or helper goroutine, and
zero warm allocations. It makes no fairness guarantee. The SIMD executor uses
the same lock for direct `ForwardTreeIntoContext` calls.

SIMD checks cancellation between layers and during recurrent/attention token
loops, plus before committing `Into` output. In-flight assembly projections
finish and all workers join before the executor unlocks. Scratch may contain
partial intermediate values after cancellation; every subsequent request
initialises all consumed rows and recurrent state. Caller output is unchanged
unless execution succeeds.

A cancellable GPU request enqueues at most 16 commands per group, synchronises,
then checks cancellation. Already launched kernels cannot be preempted. Even a
failed launch drains its possible prefix before returning. A driver failure
marks the scorer closed to inference; module/buffer cleanup retains the existing
sync-before-free and retry semantics. A background context keeps the one-batch,
one-sync path. No GPU scratch is freed or reused while its launches remain
unaccounted for.

Text scoring polls before/after each tokenizer call, before inference and before
publishing the answer. A single tokenizer call, head readout, SIMD kernel or
CUDA kernel is not interruptible. Cancellation provides no hard wall-clock
latency guarantee, especially during driver failure or a long kernel. Inputs
must remain immutable during a call. Cancellation can race with successful
completion; a request already committed before cancellation may return success.

## Verification

- Mutex tests cover canceled/nil contexts, deadlines at scorer entry, waiting
  callers, cancellation racing with acquisition, normal lock compatibility,
  concurrency and zero warm allocations. Helper statement coverage is 100%.
- Mock launch groups verify whole-group prechecks, 16-command boundaries,
  background single batch, mandatory drain after launch failure/cancellation,
  combined launch/drain errors and zero warm allocation.
- Released tests cancel at a deterministic context poll during execution on
  both backends, require nil results/context errors, then compare recovery and
  non-canceled contextual execution exactly with the ordinary API.
- SIMD `Into` cancellation preserves destination sentinels. Tokenizer-stage
  cancellation returns no answer and the same scorer/tokenizer recovers.
- Existing eight independent oracle cases, group/branch equality for 2/8/64/63
  candidates, permutations, four callers and close/use races remain enabled.
- Combined released race test passes in 225.9 seconds; whole-tree CPU race exits
  0. Focused races, vet, build and ARM64/RISC-V cross-builds pass.
- Independent cancellation review found no drain/locking/ownership defect. Its
  suspected negative-ID slice panic is prevented by the existing shared
  `validateBranchLocalText` call; a direct accelerated negative-ID regression
  now records that precondition.

No precision or GPU kernel changes occur in this pass. Oracle errors remain
`4.6759844e-5` SIMD / `4.7087669e-5` PTX logits against `3e-4`, and hidden errors
remain below `2e-3`.

## Measured cost

Same approved checkpoint and i7-12700/RTX3060/Go1.26.3 host as the
[bounded grouping pass](mojev-tree-groups-20260925.md), `GOMAXPROCS=6`.
One warmup and five full-library samples include fresh JSON decode, tokenisation,
inference and response marshal; setup and hash checking are excluded. GPU
medians with the background API versus an uncanceled cancellable context:

| Request | Background ms | Cancellable ms |
|---|---:|---:|
| Two choices | 46.07 | 45.38 |
| Eight choices | 47.63 | 48.71 |
| Two questions | 88.86 | 89.60 |
| Longer context | 92.89 | 93.10 |

Public responses are identical. These samples do not establish a speedup;
chunking introduces additional driver synchronisations. Background allocation
counts remain near the previous pass (664/1012/1034/1606 in a representative
sample). SIMD background medians were 293/399/595/768 ms, still host-sensitive.
No additional model weights or per-token buffers are allocated by cancellation.
The lock semaphore allocates once on first use, outside warmed measurements.

Evidence and benchmark probes are in `/workspace/tmp/mojev-context-20260926`.
Cancellation under these bounded paths is implemented; maximum-capacity,
long-running retention, broader concurrency admission, GPU timing attribution
and held-out quality remain open. `RuntimeReady` stays false.
