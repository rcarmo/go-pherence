## Qwen3 long-run GPU crash: software audit

Two final-evaluation attempts lost the GPU with Xid79/Xid154. Our custom runtime
remains a possible trigger; those messages do not prove hardware failure, and
short numerical parity runs do not establish long-run memory safety.

The failure locations were original64/layer34 and original677/layer9. The first
failed original subsequently completed during recovery. That argues against a
simple deterministic fault on that prompt alone, but does not rule out a
schedule-dependent or accumulated state bug. No held-out predictions or quality
metrics were consulted for this audit.

## A demonstrated context-boundary defect

The frozen encoder invokes `CopyDtoD` twice per transformer layer. That helper
called `EnsureContext()`, which pinned the goroutine, selected the CUDA context,
then released both the OS-thread pin and the driver lock. The subsequent
`cuMemcpyDtoD` call occurred outside that critical section. Go can migrate a
goroutine between those calls, while CUDA's current context is thread-local.
`MemInfo`, `SyncAll` and the native memset branch of `ZeroFloat32Buffer` had the
same pattern.

The candidate patch keeps the OS-thread pin and `cudaMu` held across context
selection and each of those driver calls, matching the existing checked upload,
download and kernel-launch wrappers. A failed memory-info query now returns zero
capacity rather than trusting output values from a failed driver call.

A model-free fake-driver regression checks lock ownership, OS-thread identity
across yields and concurrent callers. It fails on the original implementation
because the driver calls are outside the critical section, and passes after the
patch, including repeated race-detector runs. This demonstrates a violation of
the runtime's context/call invariant. **It does not demonstrate that the defect
caused either bus-loss incident.** The original test failure did not establish
an actual wrong-thread CUDA call or Xid reproduction.

## Kernel and lifetime inspection

The inspected forward path keeps the encoder lock throughout execution, retains
its mapped weights until close, uses fixed owned activation/scratch buffers and
does not free matrix scratch between asynchronous launches. The exercised
projection uses the legacy default stream and no prefix cache or CUDA graph.
BF16 widening and compensated SGEMM use checked host sizes; the inspected tile
loads/stores are masked at matrix boundaries. Qwen's 2560/4096/1024/9728 widths
and at most512 rows are well within their 32-bit element-index range. No concrete
projection out-of-bounds defect was identified by this inspection. That is not
Compute Sanitizer validation.

The current error names the layer that observed a driver failure, not necessarily
the kernel that caused it. A diagnostic-only `GO_PHERENCE_CUDA_LAUNCH_CHECK=1`
mode now synchronises after each default-stream kernel launch and reports the
function/launch dimensions on a synchronisation error. It is off by default and
skipped during graph capture; its timings are not performance measurements.

An opt-in synthetic projection guard test covers tile edges and actual Qwen
matrix widths. It checks output values and surrounding device-buffer canaries.
Compute Sanitizer2025.1.0.0 is installed under `/usr/local/cuda-12.8/bin/`.
Neither the new canary test nor sanitizer execution has run on the failed GPU.
The independent audit delegate timed out, so it supplied no corroborating review.

## Validation status and recovery sequence

Affected NVIDIA runtime/model/CLI race tests, targeted vet, host compilation,
docs/link checks and ARM64/RISC-V compile-only checks pass. The full host check
was attempted and failed in the separate Vulkan invalid-input test: this host
now returns `vulkan not initialized` where that test expects `invalid`. Do not
report a full-host pass or alter those tests to conceal the failed GPU state.

Keep this as a diagnostic candidate, not the confirmed crash fix. The original
final-evaluation executable, all676 immutable outcomes, frozen model/prompt/
calibration files and failure archives remain unchanged. No evaluation resume,
CPU substitution, GPU reset or reboot was performed during the audit.

After separately authorised GPU recovery, use a diagnostic binary and synthetic
inputs first:

```bash
GOTMPDIR=$PWD/.gotmp go test -c ./backends/nvidia/runtime -o /workspace/tmp/nvidia-audit.test
GO_PHERENCE_TEST_BF16_PROJECTION=1 GO_PHERENCE_DISABLE_NVIDIA= \
  /usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  /workspace/tmp/nvidia-audit.test \
  -test.run 'TestBF16ProjectionGPU|TestCompensatedProjectionGPU|TestCompensatedProjectionCanariesGPU'
```

Run racecheck/synccheck as separate diagnostics, then the existing synthetic
Qwen hidden/direct references, followed by a bounded variable-length synthetic
soak with persistent temperature/power/clock telemetry and kernel logs. Add
per-launch synchronisation only for attribution runs. Do not use held-out
accuracy to select a fix. A candidate that changes execution needs documented
parity and a runtime-recovery addendum before it can replace the pinned final
binary; completed records must never be silently rescored.
