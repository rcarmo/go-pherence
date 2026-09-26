# MoJev extended reuse and full SIMD admission race

The previously timed-out three-round SIMD admission race now passes with a
longer timeout. Separate ordinary runs pass 20 SIMD rounds and 60 NVIDIA rounds.
No inference implementation, numerical tolerance or concurrency limit changed.

## Workloads and execution

All runs use `84061f99`, the approved hash-checked checkpoint, Linux amd64,
Go 1.26.3, i7-12700 and `GOMAXPROCS=6`. GPU execution uses RTX 3060 with driver
580.173.02. One checkpoint/backend process runs at a time. The only test change
allows `GO_PHERENCE_MOJEV_ADMISSION_ROUNDS` from 1 to 60; the default remains 3.

Every round rejects 513-token paths and 4097-total-token requests, then scores:

- Two 512-token paths: 128 state + 128 question + 256 candidate tokens.
- A 4096-total-token request: 64 state + 64 question + 64 × 62 candidate tokens;
  each path is 190 tokens.
- The independently pinned base fixture, with the existing `3e-4` logit gate.

All successful outputs must match the warm result exactly. Each output and a
separate copy remain live across later rounds to detect mutation. These larger
stress inputs are repeatability cases; independent maximum-total parity was
qualified separately in the [grouped reference](mojev-grouped-reference-20260926.md).
Each run then checks eight simultaneous callers, cancellation while waiting,
active cancellation, recovery and post-release memory. GPU residency must stay
fixed and become zero after successful close.

## Results

| Run | Test time | Peak RSS KiB | Largest round heap increase | Heap change after output release |
|---|---:|---:|---:|---:|
| SIMD race, 3 rounds | 846.85 s | 15,339,480 | +33,248 B | −5,912 B |
| SIMD ordinary, 20 rounds | 820.86 s | 6,093,776 | +102,416 B | +50,656 B |
| NVIDIA ordinary, 60 rounds | 192.44 s | 6,057,056 | +71,560 B | −32,008 B |

Peak RSS uses `/usr/bin/time -v` from each completed process log. NVIDIA's
loaded `/proc/self/status` snapshot separately recorded 6,057,852 KiB HWM,
796 KiB above that process-log figure; these sampled counters differ slightly.

Heap changes compare post-GC snapshots with the warm baseline; the intentional
output copies and snapshot records contribute to round growth. All final
snapshots had two goroutines. GPU residency stayed at 2,057,538,816 bytes during
reuse and reached zero after close. The test retains the original CPU scorer
explicitly, so GPU host heap remains about 3.02 GB here. The separate
[host lifetime probe](mojev-gpu-host-lifetime-20260926.md) measures the savings
when that CPU scorer is dropped.

The earlier three-round race timeout was 660 seconds. The successful rerun used
1500 seconds and finished normally. A separate 60-round ordinary SIMD command
was aborted after round 59 without a final PASS or JSON report. Its partial log
is preserved as `simd-60-aborted.log`; it is not a completed qualification run.
The subsequent 20-round run completed every final check and exited zero.

These runs cover bounded repeated reuse, not hours-long service operation,
arbitrary request distributions or new native architectures. Times include
loading, GC, hashing and instrumentation and are not latency benchmarks.

## Cumulative SIMD60 closure

On `0131934d`, the unchanged ordinary SIMD admission test completed all60 rounds
in2220.96s (37m01.07s wall, exit0), closing the earlier round59 interruption.
This includes512-token paths,4096-total-token requests, over-limit rejection,
retained-output ownership, eight concurrent callers, canceled waiters, active
cancellation and deterministic recovery. The three workload score rows remain
exact throughout reuse. These stress sizes are not additional independent
accuracy fixtures; the released base-logit gate remains unchanged.

The same approved checkpoint's four asset hashes were verified. Go1.26.3,
Linux amd64, GOMAXPROCS6, NVIDIA disabled, one checkpoint process. Warm heap was
5,103,770,800 bytes; maximum retained-round heap5,103,945,088 (+174,288 bytes,
below the unchanged32MiB allowance). After releasing held outputs, heap was
5,103,838,272 (+67,472 bytes). Snapshot goroutines stayed at2, device bytes0,
and swaps0. Snapshot maximum RSS/HWM was5,003,440KiB; `/usr/bin/time` reported
5,002,880KiB. Both sources are preserved, not silently equated. The test keeps
the caller CPU scorer live deliberately; these are not packed-only heap numbers.

Evidence: `/workspace/tmp/mojev-simd60-20260926`. An initial invocation from the
repository root failed before model loading because its relative fixture path
was wrong; `wrong-cwd.log` is retained. The successful preserved test binary ran
from `model/mojev`. No implementation, tolerance or test limit was changed.
This37-minute ordinary run is **bounded reuse qualification**, not an hours-long
soak,60-round race pass, arbitrary-distribution qualification or GPU clearance.

```sh
cd model/mojev
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  GO_PHERENCE_MOJEV_ADMISSION_BACKEND=simd \
  GO_PHERENCE_MOJEV_ADMISSION_ROUNDS=60 \
  GO_PHERENCE_MOJEV_ADMISSION_REPORT=OUTPUT.json \
  /path/to/preserved/mojev.test -test.run '^TestMoJevAcceleratedAdmission$' \
  -test.v -test.count=1 -test.timeout=3000s
```

## Reproduction

```sh
# Three-round race: do not set ADMISSION_ROUNDS.
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_ADMISSION_BACKEND=simd \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  GO_PHERENCE_MOJEV_ADMISSION_REPORT=/workspace/tmp/simd-race.json \
  go test -race ./model/mojev -run '^TestMoJevAcceleratedAdmission$' \
  -v -count=1 -timeout=1500s

GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_ADMISSION_BACKEND=simd \
  GO_PHERENCE_MOJEV_ADMISSION_ROUNDS=20 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test ./model/mojev -run '^TestMoJevAcceleratedAdmission$' \
  -v -count=1 -timeout=1200s

GOMAXPROCS=6 GO_PHERENCE_MOJEV_ADMISSION_BACKEND=nvidia \
  GO_PHERENCE_MOJEV_ADMISSION_ROUNDS=60 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test ./model/mojev -run '^TestMoJevAcceleratedAdmission$' \
  -v -count=1 -timeout=360s
```

Logs and JSON snapshots: `/workspace/tmp/mojev-retention-followup-20260926`.
Focused read-only review found one evidence-label ambiguity: the peak RSS
column used process-log values while a NVIDIA snapshot was slightly higher.
The source is now explicit above; no code issue was found.

The whole-tree CPU race and builds for the unchanged inference implementation
passed with the host-lifetime slice. This test-only follow-up reruns package
races, vet/build and docs/layout checks. Native ARM64/RISC-V execution and
held-out quality/calibration are still unqualified. `RuntimeReady=false`.
