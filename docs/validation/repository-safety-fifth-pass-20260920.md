## Repository audit: failed loads, bounded helpers and descriptor ownership

A missing GLiNER tensor did not stop the loader from building the rest of its
layer stack. With a synthetic 10,000-layer configuration, rejecting the first
weight still caused roughly 360,000 boundary-loader allocations or 420,000
DeBERTa-loader allocations. The error was already known. The work continued anyway.

This pass follows the [sizing and slot-lifetime review][fourth], fixes that loading
path and checks subprocess, stream and descriptor boundaries. The [coverage
matrix][coverage] distinguishes selected source review from host tests. Neither
is a claim that every model or native backend is safe.

## Stop when the weight read fails

GLiNER's common weight reader now checks expected tensor products before asking
the source to decode them. Previously, a shape product could wrap to zero and
match an empty buffer. Boundary projection widths, floating-point FFN widths and
DeBERTa relative-embedding rows are also checked before narrowing or decoding.
The boundary and DeBERTa layer loops stop on the first read error; the shared
scorer does not enter optional loaders after its base weights fail.

Regression tests preserve the first error, check copied tensor ownership and
reject overflowing dimensions before the fake source is called. Allocation
ceilings catch continued layer construction without attempting an enormous
allocation. The existing published-shape and tiny reference fixtures pass.
These checks bound work after a failed load; they do not impose a whole-model
memory budget on valid checkpoints. Other GLiNER decoder, proposal and config
numeric paths still need review.

## Helpers and streams need aggregate limits

The shared command runner captures stdout and stderr under one byte budget,
cancels on overflow and waits for the command and pipes. Linux helpers run in an
owned process group; cancellation kills that group. Other operating systems kill
the direct child only. A two-second pipe wait prevents inherited descriptors from
keeping the caller blocked indefinitely. This does not sandbox an executable or
bound its CPU, RSS or files, and a child that deliberately escapes its process
group is outside this ownership contract.

GGUF tokenisation and the board benchmark use 4MiB capture limits; the OmniVoice
probe and TCM status helper use 64KiB. Tokenizer IDs and benchmark JSON rows are
validated instead of accepting partial or unrecognised output. Tests use fake
helper processes and captured fixtures, not vendor model execution.

The serving benchmark now bounds the complete response as well as individual
SSE frames. Defaults are 64MiB per response and 1,048,576 retained content events;
requests and concurrency are capped at 1,048,576 and 65,536 respectively. Usage
counts must be nonnegative and internally consistent. Aggregate counts saturate
rather than wrap, and percentile arithmetic handles NaN and wide duration spans. These
are per-request/workload limits, not a process-wide memory budget. Callers still
need suitable concurrency and deadlines.

## File descriptors are not GPU teardown

The experimental NVIDIA ioctl path now owns all five device descriptors, closes
partial opens and later initialisation failures, handles descriptor zero, and
retains a successful CPU-mapping descriptor until its buffer is freed. Host
handle/VA counters are serialised and handle exhaustion is rejected. Injected
open/close functions and ordinary temporary files test those rules without
opening NVIDIA devices or issuing ioctls.

This does not resolve GPFIFO notifier ownership, failed channel cleanup, work
submission/drain, concurrent mapped reads/free or device Close during execution.
Setup and Close require a quiescent owner. CUDA capture ownership and shared
scratch, together with the GGML/ORT native lifecycle findings, remain open.

## Rejected SIMD inputs still reach scalar code

SIMD dispatch could reject a huge RoPE position only for the fallback to wrap its
frequency offset. The scalar path now bounds the starting pair before indexing,
preserving its existing partial-frequency and clamped-head behaviour. The
allocation-returning attention helper validates dimensions, backing extents,
callbacks and scale before allocating. Native CPU-dispatch and feature-disabled
regressions pass; no GPU attention was executed.

Whole-buffer audio resampling rejects invalid rates, overflowing sizes and output
above 1GiB before allocation. Output length uses checked quotient/remainder
arithmetic instead of a floating-point conversion. Valid constant-signal and
length tests pass alongside the audio, MOSS and Whisper host tests. Large or
untrusted media still needs the tighter streaming admission in `loader/audio/media`.

## New findings left open

Direct synthetic checks of Qwen3-TTS's embedding, FFN and attention layouts found
the same wrapped-sizing pattern previously fixed in LFM2. Matching wrapped counts
can pass validation; extreme KV sizing returned `-2` floats or `-8` bytes with a
nil error, and NaN norm epsilon was accepted. These are planning-layout defects,
not measured model execution. They are not fixed in this pass. Adjacent Qwen3-TTS
config, projection and runtime contracts need a coordinated arithmetic review.
The attempted delegated review timed out; these findings came from direct source
inspection and a small host-only reproducer.

Source-only SpacemiT review also found that TCM accessors can retain pointers after
Close, the reference count does not prevent unmapping, and the native C shim
checks only nonempty buffers before narrowing dimensions and passing pointers.
The public config flags are mutable globals, so concurrent mutation is not a
supported configuration protocol. No K3 device or native shim was executed.

## Checks and preserved state

The final NVIDIA-disabled race sweep on committed code exits zero: **102 packages
pass and 61 have no tests**. It includes the RoPE, audio and GLiNER fixes, unlike
the earlier fifth sweep retained in the logs. Whole-tree vet/build, Linux ARM64
and RISC-V cross-builds, 22 Bun tests and the 334-document link/layout check pass.
Foreign builds are compilation only, and host package success can include skipped
hardware or unavailable fixtures. Selected-source coverage is now **82/163**
packages; **81 packages remain inventory/tests-only**.
Logs and synthetic reproductions are retained under
`/workspace/tmp/go-pherence-repo-audit/`.

Both preserved evaluation executables, all 16 freeze-manifest entries and all
676 record hashes match. Hash checks read bytes, not prediction quality. The
four-arm evaluation is still blocked at 676/1,440; no completed row was replayed,
no weights or calibration changed, and no GPU recovery or service restart was
attempted. The audit remains open while the unchecked source and native ownership
findings are worked through.

[fourth]: repository-safety-fourth-pass-20260919.md
[coverage]: repository-safety-audit-coverage-20260919.md
