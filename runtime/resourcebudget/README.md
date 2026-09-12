# Cooperative CPU and memory reservations

`resourcebudget` atomically reserves declared CPU slots and estimated memory for cooperating owners in one process. It does not start workers, change `GOMAXPROCS`, measure RSS, set OS limits or contact other services. Share the **same Budget instance** across owners; separate instances and processes do not coordinate.

## Use

```go
b, err := resourcebudget.New(resourcebudget.Config{
    Capacity: resourcebudget.Resources{CPUSlots: 4, MemoryBytes: 6 << 30},
    MaxActive: 8,
    MaxWaiting: 32,
})
if err != nil { return err }
lease, err := b.Acquire(ctx, resourcebudget.Resources{CPUSlots: 2, MemoryBytes: 2 << 30})
if err != nil { return err }
defer lease.Release() // only after all owned work has drained
```

A reservation can have zero CPU slots or zero memory bytes, but cannot be empty or negative. Requests above total capacity fail immediately. Both dimensions and the active-lease count are checked under one mutex, with subtraction-based bounds to avoid overflow. A waiting caller holds neither partial CPU nor partial memory capacity.

Pending requests are FIFO by mutex registration order. A smaller request cannot bypass a head that does not fit. This can leave capacity idle; there is no priority, preemption or deadline scheduling. Cancellation removes pending requests and wakes eligible successors. Pending entries and active leases have independent configurable bounds (at most 65,536 each); a full pending list returns `ErrBusy`. A zero waiting bound permits immediate acquisition only.

`Acquire` returns a pointer to a lease; **do not copy the lease value**. Like `sync.Mutex`, it contains a no-copy marker checked by `go vet`. `Release` is idempotent on that lease and clears its accounting. Cancellation after a successful grant never revokes it. Owners must stop their work, join CPU/subprocess/device activity and then release. A granted reservation racing a cancelled wait is returned to the budget before cancellation is reported.

`Shrink` only reduces an active lease without releasing/reacquiring its retained dimensions. For model loading, drop CPU slots and temporary loading bytes while retaining resident memory. Use `Release` for zero demand. Do not acquire a second lease whose demand cannot fit alongside a lease you already hold; that owner-level dependency can deadlock. The allocator does not infer dependencies or manage model disposal.

`Admission(demand)` validates and returns a fixed callback compatible with `speechjob.Admission`. Callback construction reserves nothing. Use it for a queue or the HTTP handler's synchronous `RunAdmission`. Different callbacks from one budget share counters and waiting order. Caller estimates include all relevant worker, native, mapping, activation and subprocess use; understated estimates invalidate the operator's safety assumptions.

## Shutdown and observability

`Snapshot` returns detached scalar capacity/usage, active/waiting counts and closed state. These are accounting values, not host utilisation or RSS samples.

`Close` stops admission and wakes pending callers with `ErrClosed`. It neither cancels active work nor releases its reservations. `Shutdown(ctx)` also waits for all active owners to call `Release`. Timeout leaves leases accounted and is not permission to free live resources. In-process state disappears on process death; there are no persisted leases or cross-process lock protocol.

One lease allocation is used on uncontended acquisition; snapshots allocate nothing in the regression tests. Waiting calls allocate a bounded entry/channel; no allocator goroutine, timer, background sampler, object pool or model cache exists. Mass cancellation and batch grants compact the pointer list in linear time and clear removed references. Mutex waiting and arbitrary owner callbacks are not preemptible.

## Speech server integration

The [explicit server](../../cmd/audio/speechjobserve/README.md) accepts optional `resources` estimates. It reserves loading CPU/memory before `buildProfile`, shrinks to resident memory after loading, and retains resident accounting through handler drain. Transient work reserves additional CPU/memory for queue or synchronous execution. Its private budget has two active slots (resident plus one serial job); it cannot coordinate a separate LLM process.

The existing token-prefill/decode scheduler is unchanged. Tests simulate cooperating non-speech/LLM demand without loading or controlling a real LLM. Host-available-memory checks, hard RSS/CPU cgroups, external service-state coordination, trained peak estimates and GPU budgeting are unfinished.

## Verification and refinement

Run `make speech-job-check` or `go test ./runtime/resourcebudget`. Checks cover multidimensional atomicity, FIFO, active/wait bounds, cancellation/grant races, shrink, max-integer capacities, idempotent release, close/drain and allocation/retention bounds. HTTP tests share one budget across synchronous/queued stores with simulated non-speech demand; cancellation retains active ownership until drain. Server tests use generated toy assets and explicit FFmpeg fixtures only.

1. Problem: serial exclusion does not account for shared declared CPU/memory capacity.
2. User: Go owners embedding recording jobs and other cooperative inference work.
3. Success: no partial/oversubscribed accounting; bounded FIFO waits and explicit drain.
4. Minimum: process-local multidimensional lease, status and callback bridge.
5. Exclusions: real LLM control, OS enforcement, priority/preemption, dynamic sampling and trained sizing.
6. Scope: new runtime package; additive HTTP callback and opt-in server estimates.
7. Platforms: standard-library Go; scoped cross-builds, Linux execution tests.
8. Errors: reject invalid/oversized/wait-list-full; shutdown wakes pending only.
9. Persistence: none; durable queue remains independent.
10. Pattern: owner-held leases through drained work, not token scheduling changes.
11. First change: allocator invariants/tests before integration.
12. Avoid: generic distributed scheduler, nested partial reservations, global process settings and pools.
13. Names: Resources, Budget, Lease, Snapshot; memory values are bytes.
14. IO: explicit capacity/demand/context; lease or stable error, detached counters.
15. Compatibility: existing server defaults unchanged when resources are absent.
16. Limits: bounded active/waiting counts, integer-safe counters, linear cleanup; one fast-path allocation.
17. Trust: cooperative same-instance callers; lease values non-copyable; estimates operator-owned.
18. Export: status counters and reproducible source/test evidence; no host secrets or model data.
19. Proof: repeated invariant/concurrency tests, HTTP drain integration, toy server and process-signal checks.
20. Closure: scoped tests/vet/build/commit/evidence; external coordination and full speech gates stay open.
