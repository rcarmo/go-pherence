# MoJev GPU bus-loss audit

The failure is confirmed as NVIDIA Xid79 (GPU fell off the PCIe bus), followed
by Xid154 (node reboot required). Source and generated-PTX inspection has not
identified a computation-kernel bug that explains it. The failure is unresolved;
GPU stress testing and adoption of the retained candidate remain blocked.

## Incident and corrected attribution

The guest journal records both Xids at `2026-09-26T11:21:52+01:00`, the same
second as the failing test log. The machine is a QEMU guest; device is RTX3060
at guest PCI `0000:00:10.0`, driver580.173.02. CUDA12.8 / Compute Sanitizer2025.1
were used. No GPU reset, module unload, guest/host reboot or unrelated service
change was performed during this audit.

The retained failure is:

```text
TestMoJevPTXTreeKernels/capacity512/warp_attention
nvidia_tree_kernels_test.go:192: cuMemcpyDtoH: error 719
RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

Line192 calls `run(ref, ids, false)`, where `ref` is `mj_attention`. It launches
the old compact reference kernel after the new tree-warp output has been
computed. The outer test name is not the kernel name. The failure was observed
on a reference download; asynchronous/sticky CUDA errors still prevent assigning
causality to that reference or excluding earlier work. The test helper captured
the parent `*testing.T`, causing an additional parent-FailNow warning. That is a
real diagnostic defect, not an identified cause of GPU bus loss.

The earlier description that the new warp kernel itself failed after60.7s was
too specific. That duration belongs to the subtest, which runs multiple kernels.
CUDA719 means launch failed; CUDA702 is the distinct launch-timeout code. Neither
719 nor the duration proves a watchdog timeout. Zero displayed race hazards with
a failed target is not a sanitizer pass.

## Source, PTX and host-runtime checks

Exact candidate: `a3fffac4`. The separately reviewed preserved warp/preparation
sources were verified to match the candidate's new kernel blocks.

- **Warp attention:** 256-thread blocks, eight complete warps; grid size equals
  row count. Query/head and K/V/output offsets stay within the fixed strides.
  Tree start/end metadata is validated before upload. Warp control flow is
  uniform for each shuffle group, including early exit; full-mask shuffles are
  valid under this launch. No block barriers or shared state are used.
- **Prepared recurrence:** one prepare thread per token/head writes disjoint
  alpha/beta slots. Preparation precedes recurrence in the same stream, and
  projections overwrite those slots on subsequent layers/replays. The original
  fork-state rule and per-warp row ownership are unchanged.
- **Reference attention/reductions:** the compact reference uses the existing
  block reduction and barriers. Bounds/control flow are uniform under the tested
  geometry. No missing barrier or concrete buffer overrun was found.
- **Generated code:** offline `nvcc -cubin -arch=sm_86 --fmad=false` compilation
  reports48 registers and zero stack/spill/shared-barrier use for warp attention,
  37 registers/no spills for prepared recurrence, and16 for parameter prep.
  This examines uninstrumented code; sanitizer instrumentation can differ.
- **CUDA driver scope:** upload/download/free, launch and sync hold the OS-thread
  pin and driver mutex across context selection and the call. Previous fix
  `490b3d9b` is in main ancestry. Current model-free driver-scope/DtoD/launch
  regressions pass under race three times. The previously found thread-migration
  defect is not reproduced in this path.

Independent read-only reviews found no substantiated source-level indexing,
shuffle, synchronization or ownership defect under these invariants. They do not
prove device safety or exclude application code as a trigger.

## Recurrence and limits of the evidence

Prior evaluator incidents recorded the same Xid79 → Xid154 → CUDA719 signature
before these new MoJev kernels existed. Earlier investigation found the real
CUDA context bug subsequently fixed in `490b3d9b`, but did not prove that bug
caused bus loss. The current recurrence means that fix cannot be treated as a
complete explanation or resolution of all GPU failures.

The combined candidate passed ordinary and race-instrumented released short,
512-path and4096-total reference tests, expanded CUDA memcheck, and small-fixture
racecheck. The maximum-capacity racecheck failed. Main retains the previous
GPU dispatch; candidate `a3fffac4` and diagnostic follow-up `a8b52e4b` are on
`experiments/mojev-gpu-combined`.

GPU temperatures47–65°C were sampled in the preceding benchmark matrix, not at
the failure. No continuous GPU power/temperature/Xid collector covered the final
sanitizer run. This gap prevents ruling out heat/power or correlating load at the
incident. Later `nvidia_dev_put` warnings attributed to MiniBrowser occur after
the Xid; they do not show that MiniBrowser caused it. Physical-host PCIe/VFIO,
power and temperature evidence for this incident has not been collected in this
audit. Earlier clean host logs must not be reused as evidence for this event.

A bounded `nvidia-bug-report.sh --safe-mode` attempt timed out and left a partial,
readable diagnostic archive. Remaining processes from that diagnostic collection
were terminated; no unrelated process was stopped. The archive is kept locally
because it can contain host details. It is incomplete, not a finished NVIDIA
crash report.

## Changes and next safe investigation

A separately gated [single-launch attention probe](mojev-attention-isolated-probe-20260926.md)
is now available for the compact and tree kernels. Its default is three rows,
with a CPU oracle and guarded output; no comparison GPU kernel is launched.
Only model-free tests and compilation have run. This does not authorise GPU
execution or close the device-loss investigation.

The candidate test now passes the correct subtest `*testing.T`, logs actual
kernel name/tree mode/rows/grid/block, and explicitly synchronizes each prepare
and tested launch before downloading. This narrows future asynchronous-error
attribution. These diagnostic-only changes compile and pass model-free
races/vet/docs checks; they have not been run on GPU since the fault.

Before further GPU execution:

1. Preserve guest and physical-host logs/crash state; authorised recovery is
   separate from diagnosis. A reboot alone does not establish a fix.
2. Arrange continuous guest/host PCIe/Xid and GPU temperature/power/clock logs
   through the whole test, not only performance sampling.
3. On a recovered device, isolate the compact reference and candidate kernels
   in separate small processes with explicit completion fences. Increase shapes
   deliberately only after stable baseline results. Stop on the first driver
   error/device-query failure; do not automatically retry or reset.
4. Compare ordinary, memcheck and racecheck behaviour. Do not repeat the full
   maximum-capacity sanitizer suite blindly, relax numerical gates, or declare
   a zero-hazard failed run clean.

Possible causes still include an application-triggered driver/device fault,
sanitizer/driver interaction, physical PCIe/power behaviour and passthrough.
No one of these is established. Evidence is under
`/workspace/tmp/mojev-gpu-audit-20260926` and the prior
`/workspace/tmp/mojev-gpu-combined-20260926`. `RuntimeReady=false`.
