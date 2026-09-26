# GPU state guard

`scripts/gpu-guard.ts` records GPU state throughout a bounded run and stops its
owned workload on missing telemetry, driver faults or unsafe temperature. It
runs **monitor-only by default**. No GPU kernel or model was executed while
qualifying this harness after the September26 bus-loss incident.

## Behaviour

Before starting an explicitly authorised command, the guard:

- Reads the current boot's kernel journal and refuses any recorded Xid, GPU
  bus-loss/recovery action, or NVRM fatal/error message. Old-boot logs are not
  mistaken for current faults; an Xid in the current boot is not ignored merely
  because device queries later recover.
- Requires one unambiguous GPU (or an explicit UUID), valid temperature/utilisation/
  memory data, and no other compute process on the selected device.
- Pins its expectation of boot ID, GPU UUID, PCI address and driver version.
- Requires writable local evidence files and readable kernel-journal access.

During execution it records timestamped temperature, utilisation, used/total
memory, power and clocks plus compute processes and kernel-journal progress.
Power/clocks may be explicitly unavailable on unsupported drivers; required
fields cannot silently become zero or N/A. New journal messages are read by
cursor. Queries have a deadline and output cap; failed, malformed, warning or
permission-denied responses stop work rather than trigger retries. The default
sample interval is1s and each query timeout2s; sequential queries and termination
grace mean detection/stop is not instantaneous.

Boot, UUID, PCI or driver discontinuity, a fresh Xid, missing GPU, competing
compute process, >=83°C (or a lower configured limit), child failure, total
workload deadline, signal or evidence-write failure makes the run fail. A final
journal/health/journal check runs even for a quickly successful child. No
automatic rerun, GPU reset, service restart or device recovery exists.

An owned supervisor stays alive after the command exits, preventing its PGID
from being recycled before teardown. The supervisor signals its own group with
TERM and KILL after750ms, and the parent verifies no live members remain.
Parent-pipe loss also stops the group. This handles ordinary descendants, not
commands that deliberately escape the session using `setsid`, nor remote
workers. Do not use it for daemonising commands. Unkillable driver processes
can still require manual inspection; a timeout is not guaranteed device recovery.

## Evidence and use

Each run requires a new output directory, created with mode0700. It writes
mode0600 `events.jsonl`, `result.json`, and (only in command mode)
`workload.stdout`/`workload.stderr`. Every event is flushed with `fsync`; final
result is flushed too. Write failures fail closed and print remaining evidence
to stderr/stdout if files are unavailable. Use persistent workspace storage,
not `/dev/shm`, and avoid secrets in command arguments because options are logged.
GPU/journal binaries use explicit `/usr/bin` paths; arbitrary workload arguments
are passed without a shell. Run in a trusted environment, not as a privilege
boundary against hostile commands or environment manipulation.

Monitor-only (no child process):

```sh
bun scripts/gpu-guard.ts --out /workspace/tmp/gpu-monitor-NEW --seconds 10
```

Authorised foreground command (this flag is not itself authorisation):

```sh
bun scripts/gpu-guard.ts --out /workspace/tmp/gpu-run-NEW \
  --seconds 30 --run-authorized -- /absolute/path/to/approved-command args
```

Optional settings: `--uuid GPU-...`, `--interval-ms 250..5000`,
`--query-ms 100..5000`, `--max-temperature 40..83`. Runtime defaults to30s and
is bounded to1..3600s. Longer or broad GPU stress runs still require separate
approval; the frozen evaluation schedule and model services are not resumed.

The guest journal and GPU queries do not collect physical-host PCIe/VFIO logs,
PSU rail data or every graphics workload. `--query-compute-apps` is not a complete
GPU-consumer inventory. Coordinate physical-host logging separately during an
actual device-fault investigation. No numeric temperature threshold proves
hardware safety or explains the prior Xid79 incidents.

## Qualification

CPU-only fake-I/O tests exercise query failure/timeouts, malformed data, initial
and fresh Xids, missing/replaced GPU, driver/boot changes, competing processes,
child failure, deadlines, fast-child final checks and logging failures. Real
CPU-process fixtures test normal/nonzero exits, failed spawn, TERM-ignoring
children/grandchildren, idempotent stop and parent-pipe disappearance. No test
uses GPU computation. Run all suites with:

```sh
bun test scripts/gpu-guard*.test.ts
```

Source review caught unsafe raw post-exit PGID signalling and insufficient
leader-only cleanup verification. Those were replaced with the stable supervisor
and explicit group checks. It also prompted absolute query paths and fail-closed
logging. Follow-up review caught premature ready reporting on failed spawn,
which now rejects startup. One intermediate review timed out and is not counted
as completed coverage.

The first real monitor attempt stopped in preflight on an invalid journal option
(`--kernel`); it preserved failure evidence and launched nothing. After switching
to supported `-k`, two10-second monitor-only checks each passed with12 samples.
Observed boot `a6949e95-2bb5-4e1b-b01d-bde18be7d7d2`, RTX3060 UUID
`GPU-cb9db783-7465-139d-900c-bc8d6b6ced7c`, PCI `00000000:00:10.0`, driver
`610.57.04`. Initial temperature54°C, memory33/12288MiB, utilisation0%; no new
kernel fault observed. A separate `/usr/bin/true` command run passed final checks
and group cleanup; it performed no CUDA work. These checks establish monitor
operation, not sustained GPU stability or a fix for bus loss.

Raw evidence: `/workspace/tmp/mojev-gpu-guard-20260926`. The GPU isolated-kernel
probe and retained optimisation candidate remain hardware-unverified on this
new driver until separately cleared and executed. `RuntimeReady=false`.
