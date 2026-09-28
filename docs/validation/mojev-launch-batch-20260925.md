# MoJev GPU launch allocation pass

A bounded launch batch reduces Go allocation churn without changing PTX kernels
or arithmetic. This follows the [SIMD allocation pass](mojev-accelerated-text-20260925.md)
shipped as `3860f490` (CI `36196903094` passed).

The GPU allocation profile attributed 53.8% of objects to the reflective FFI
binding; CUDA launch/context plumbing was the main caller. `LaunchBatch` stores
scalar argument bits in caller-owned fixed storage, validates every command
before execution, and pins the driver thread/context once per branch. Up to 512
commands are reserved at scorer construction. Input/output staging also uses
bounded host buffers, consumed under the scorer mutex.

The driver reads host parameters synchronously at enqueue. Arguments remain
stable throughout the call. The API follows the current capture/default stream,
keeps launch counters and diagnostic sync behaviour, and returns driver errors
without retrying kernels. A failed batch can have launched a prefix. Scorer
`Close` calls `PTXModule.Close`, which synchronises before unloading or freeing
buffers; failed sync/unload preserves ownership for retry.

## Five-sample results

Same host, pinned checkpoint and four full-request workloads as the preceding
record; one warmup, five samples, `GOMAXPROCS=6`, sequential processes. Before is
the saved GPU binary from the preceding allocation pass. All four saved example
responses are byte-equivalent after JSON comparison. The dedicated eight-case
fixture remains within its unchanged gates.

| Request | Before ms | Batch ms | Allocations before → batch | Bytes before → batch |
|---|---:|---:|---:|---:|
| Two choices | 90.15 | 91.32 | 13,257 → 9,431 | 1,056,320 → 327,136 |
| Eight choices | 366.57 | 367.18 | 51,498 → 36,193 | 4,964,840 → 1,261,720 |
| Two questions | 177.17 | 177.85 | 26,220 → 18,568 | 2,036,616 → 643,800 |
| Longer context | 184.95 | 184.36 | 14,199 → 10,373 | 2,751,184 → 367,232 |

Medians show 27–30% fewer objects and 69–87% fewer allocated bytes. Latency is
essentially unchanged. GPU device residency remains 2,024,992,000 bytes at
capacity 256; the new reusable host staging costs 2 MiB plus fixed command
storage. Reflective driver calls still allocate. No zero-allocation GPU runtime
or general speedup is established.

## Verification

- Whole-tree CPU race suite: exit 0, `batch-whole-race.log`.
- Released PTX race test: pass, eight cases, exact isolation and repeated calls;
  maximum logit error `4.7087669e-5`, hidden error `7.8797340e-4`.
- Kernel tests exercise direct and batched GEMM, including tile tails. NVIDIA
  compute-sanitizer memcheck reports zero errors; racecheck reports zero hazards.
- Mock-driver tests cover whole-batch prevalidation, lock scope, scalar argument
  bits, default/capture streams, diagnostic sync and driver failure. The mocked
  warm path asserts zero Go allocations; actual purego calls still allocate.
- `go vet ./...`, `go build ./...`, ARM64/RISC-V cross-builds and diff checks pass.
  Native hardware execution here is NVIDIA/amd64 only.
- Focused independent review found no ABI argument issue. Its proposed missing
  close synchronisation was checked against `PTXModule.Close`: the existing
  synchronisation occurs there, before any resource release. A comment now makes
  that dependency explicit.

Raw samples, profile, commands and logs are under
`/workspace/tmp/mojev-optim-20260925`. Candidate ancestor reuse, CUDA timing
attribution, deeper FFI allocation removal, cancellation and full admission
qualification remain open. `RuntimeReady` stays false.
