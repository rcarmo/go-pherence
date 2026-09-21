## Repository audit: planning arithmetic and native entry points

Later update: the [seventh pass](repository-safety-seventh-pass-20260920.md) fixes
the mmap advisor and FP16 findings below and adds host-tested LLaMA config checks.
Native allocation/upload/decode lifecycle validation remains open. This report
preserves the sixth-pass results.

Qwen3-TTS's planning structs could return negative KV byte counts without an
error. The embedding and FFN validators also accepted dimensions whose products
wrapped, provided the caller supplied the same wrapped totals. Those defects
were reproduced in the [fifth pass][fifth]; this pass fixes them and follows the
same failure pattern into packed-Q4 wrappers and compressed KV storage.

The [coverage matrix][coverage] records selected source sections, including new
findings that remain open. The frozen four-arm evaluation has not resumed.

## Checked counts all the way to the request

Qwen3-TTS now checks projection, embedding, FFN, KV, prefill and fused-input
products and sums. Validators reject invalid intermediate counts before comparing
them with declared values, so matching wrapped numbers cannot establish validity.
Byte calculations check int64 overflow separately. Decoder code counts, waveform
and reference-audio sample counts use checked products; seconds are checked for
nonfinite values and range before conversion to frames. NaN or infinity cannot
pass the RoPE/norm controls through ordinary inequality checks.

The config parser preserves integers beyond float64's exact range and rejects
malformed *present* numeric fields rather than truncating them or silently using
defaults. Missing fields retain their existing defaults. Known nested sections
must be objects, integer fields use in-range decimal integer syntax, and MRoPE
sections require three nonnegative integers. Materialised code-group lists are
limited to 1,024 entries; published checkpoints use 16. This bounds that metadata
allocation, not the memory footprint of an arbitrary valid model.

Tensor inspection now distinguishes nested `talker.code_predictor` weights from
Talker weights. A regression uses different widths for the two stacks, since equal
widths concealed the mistake. Projection checks compare both matrix axes, including
when the expected dimensions are equal. Inspection validates its config first.
Unknown tensor names remain outside these selected shape checks.

The baseline regressions failed on wrapped counts, NaN controls, rounded integers,
malformed nested objects and misclassified predictor matrices. The existing small
fixtures and inspector tests pass with the fixes. Qwen3-TTS is still a planning
scaffold: the default execution interfaces return `ErrRuntimeNotImplemented`.
A requested independent review failed at workspace-path resolution and supplies
no additional review coverage.

## Before C or assembly sees a pointer

The experimental packed-Q4 C and Plan 9 wrappers previously multiplied untrusted
block/group counts in their length checks, rounded dimensions with unchecked
addition, and accepted nil tile destinations. The shared extent checker now
validates those products, output byte spans and padded row ranges before native
entry. Signed C/translated-kernel counters leave room for rounding and loop
increments. The native kernels require F16C for their half-precision scales;
dispatch now checks it explicitly through the existing CPU capability helper.

Malformed-call tests exercise both available wrappers and unsupported stubs
without passing invalid buffers into native code. Existing synthetic CPU tile and
projection parity tests pass on this host, and feature-disabled tests return
unsupported errors. Four missing non-amd64 tile stubs were added; ARM64/RISC-V
test compilation verifies their API, not their execution. This review covers
wrapper geometry and selected C orchestration, not every translated assembly
instruction or every CPU profile.

## NaNs must remain NaNs

BF16 narrowing rounded every float32 bit pattern as though it were finite. A NaN
with a small payload became infinity; other payloads could wrap the exponent and
sign. The conversion now lives in `half`, quiets NaNs before rounding, and is used
by the BF16 backend. Exhaustive BF16 round trips and explicit ties, signed zero,
infinity and payload tests preserve the existing finite round-to-nearest-even
behaviour. The separate float32-to-FP16 converter was not changed; its nonfinite
conversion policy needs a follow-up review.

The checked and void BF16 RMSNorm paths also reject negative and nonfinite epsilon
before writing. Zero retains the existing no-regularisation behaviour, so callers
must not assume it produces finite output for a zero vector. No model-quality
claim follows from these representation and input checks.

## Compressed storage is still mutable storage

Appending an extra entry to the compressed KV cache's exported storage could make
GetK slice beyond its scratch buffer. Count checks now verify paired K/V storage
and the total token count before append or decompression; head geometry and packed
byte products are checked as well. Invalid append leaves the cache unchanged.
The legacy accessor fallback returns full storage when compressed state is invalid,
so callers must still check the expected length rather than treating fallback as a
complete reconstruction.

Reset now clears entry pointers before shortening compressed slices, releasing
references to old payloads. Full and scratch backing capacity can still be retained.
The cache's byte reports describe logical payload, not retained capacity or RSS.
Access and lifecycle require external serialisation, and returned full/scratch
slices are borrowed. The baseline GetK panic is retained in the local test log;
KV and model integration race tests pass after the fix.

## Source-only findings and small helpers

The mmap advisor keeps a borrowed mapping and cannot prevent its owner from
unmapping it. Its cold-eviction snapshot can become stale after another goroutine
touches a range, and overlapping tracked ranges can overcount bytes. Those hints
and statistics are not an ownership lease or a hard residency budget.

The native LLaMA GGML wrapper checks an upper layer limit but indexes config arrays
and narrows dimensions before validating their full lengths/ranges. Calls after
Close and concurrent setters/decode/Close need an owner protocol, while the host
stub lacks several native API fields/methods. The required GGML build is unavailable
here; these are source findings, not a tested repair. Existing CUDA capture/scratch,
GPFIFO, ORT and TCM/native-shim findings remain open.

Selected CLI-helper review covered prompt-file loading, path basenames, Whisper
and K3 environment flags, and DiffusionGemma budget parsing. Prompt files close on
error and retain Scanner's per-line bound, but have no aggregate prompt limit.
Environment flags are process-start policy, not a per-request model switch.
DiffusionGemma's MiB conversion now saturates instead of wrapping signed budgets
and clamps negative defaults to zero. Saturation does not authorise allocation of
that amount. Namespace-only backend packages and the GGML island planner were
also read; neither establishes native backend correctness.

## Checks and preserved state

The fresh committed-code NVIDIA-disabled race sweep exits zero: **105 packages
pass and 59 have no tests**. The earlier sixth sweep passed 104 packages, with 60
reporting no tests, but started before the final cache, flag and dispatch changes
and is retained separately. Targeted regressions, whole-tree vet/build, Linux
ARM64/RISC-V builds, 22 Bun tests and the 335-document link/layout check pass.
Qwen3-TTS and Plan 9 Q4 foreign test binaries compile; none was executed on a
foreign target. Selected-source coverage is **97/164 packages**, leaving **67
inventory/tests-only**; selected sections are not whole-package clearance. Hardware/fixture skips inside a passing host package remain
possible. Local logs and reproducers are under
`/workspace/tmp/go-pherence-repo-audit/`.

Both preserved evaluation executables, all 16 freeze-manifest entries and all 676
record hashes match. No predictions were scored or inspected, no completed row
was replayed, and no GPU recovery or service restart was attempted. The remaining
native ownership work and unreviewed packages keep the audit open.

[fifth]: repository-safety-fifth-pass-20260920.md
[coverage]: repository-safety-audit-coverage-20260919.md
