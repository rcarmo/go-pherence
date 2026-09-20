## Repository audit: sizing, slot lifetime and diagnostic results

A planning validator is not safe if it compares two equally wrapped integers.
LFM2's remaining sizing formulas had that problem, while the expert-stream API
returned mapped slot views with no way to retain ownership through a consumer.
This pass fixes those boundaries and follows several false-success paths into
the benchmark and inspection tools.

This continues the [third pass][third]. The [coverage matrix][coverage] lists
selected source sections, not whole-package clearance. The frozen evaluation
remains stopped at 676/1,440, with no partial quality inspection, completed-row
replay, new weights, GPU recovery or hardware restart.

## LFM2 layout arithmetic

Convolution state/weights, attention KV/projections, router, embeddings, FFN,
normalisation and stage-contract sizes now use checked nonnegative products and
sums. Constructors validate before returning; forged layouts cannot pass by
supplying the same wrapped size that the validator computes. Byte counts and
per-token scratch return errors on overflow. Norm epsilon, RoPE theta and router
scale reject NaN/Inf rather than passing ordinary inequality tests.

Schedule and execution-plan index lists must now match their actual steps, not
merely have the right length. Constructors that accept a supplied schedule or
execution plan validate it before using its layer counts. Tests cover extreme
sizes, declared wrapped counts, byte/scratch overflow, nonfinite controls and
incorrect list membership. These are metadata/planning checks. LFM2 inference
remains unimplemented and arbitrary configuration memory admission is not an
OS-enforced resource budget.

## Slot ownership spans the consumer

`expertstream.Reader.WithExperts` holds ownership through a synchronous callback,
excluding other loads and Close until the consumer returns. The experimental
model upload adapter prefers this scoped API when supported; synchronous host
copies finish before slot ownership is released. No GPU transfer was executed
here. Legacy injected sources and the existing Load API retain their external
serialization contract.

Callbacks must not re-enter the reader, retain views or leave background/native
consumers running after return. Deferred unlock handles errors and panics, but
does not wait for work that the callback incorrectly abandons. Tests block a
callback while attempting slot reuse or Close and check error/panic lock release.

Open now checks aggregate slot-mapping bytes, including alignment slack and OS
page rounding, before mmap allocation. Zero MaxSlotBytes selects a 1GiB default;
explicit positive budgets allow larger workloads. Slots are limited to the expert
inventory and 65,536. Existing worker configuration semantics are preserved;
actual read-worker count was already capped by assignments. This bounds anonymous
slot mappings, not manifest/file-cache/Go-heap RSS. Manifests and paths remain
trusted local inputs, and unscoped raw views still cannot outlive reuse/Close.

## Benchmark success must mean a complete response

The SSE benchmark parser now requires `[DONE]`, stops there and rejects streamed
error objects. HTTP 200 with an empty/truncated response no longer counts as a
successful request. A 4MiB line/frame bound prevents unbounded allocation from an
unterminated event, while preserving the existing 70KB payload test. Retained
frame strings are cleared between events. This bounds each frame, not the whole
stream: callers still need a timeout and workload admission.

Arrival scheduling rejects NaN/Inf or elapsed nanoseconds outside time.Duration
before narrowing. Run rejects a nil context and starts no more workers than
requests. A zero request timeout still deliberately means caller-controlled
lifetime; usage-counter validation and arbitrary benchmark-size admission remain
separate concerns.

## SIMD fallback and fixture accounting

The board fallback's parallel GEMV sliced rows before checking matrix extents.
It now validates dimensions and full buffers first. Attention validates GQA
head grouping, sizes and scale before slicing, and uses the existing SIMD dot
kernel rather than its scalar inner loop. Independent float64-oracle, tail and
malformed-dimension tests pass with CPU features both enabled and disabled.
This does not exercise Vulkan or K3 hardware.

Fixture tensor summaries now hash canonical little-endian float32 bytes through
fixed 1KiB scratch instead of allocating a 4*N byte copy. Existing checksums and a
multi-block independent hash oracle match. NaN/Inf inputs return errors rather
than producing invalid JSON statistics. On a synthetic 1,048,576-element tensor,
retained benchmark samples changed from 4,194,988--4,195,043 B/op and 8--9
allocations to 214 B/op and six allocations. Timing ranges overlap (3.60--4.71ms
before, 4.39--4.56ms after), so the result is allocation reduction, not a speedup.

## Inspectors must not hide broken inputs

Optional safetensors metadata now omits only genuinely absent default weights.
Explicit missing paths, dangling indexes, missing shards and corrupt present
files remain errors. LFM2 and Qwen3-TTS inspectors use that distinction instead of
silently suppressing every resolver failure. Their documented metadata-only mode
and strict semantics are retained; absent weights are not advertised as runtime
readiness.

The small `embcheck` and `shapecheck` diagnostics also ignored argument, open and
dequantisation errors before indexing fixed token/width ranges. They now validate
arguments, close files, propagate errors and bound printed samples. Raw Q6_K block
inspection runs only for that dtype with a complete block. Tests use malformed
files and a one-element synthetic GGUF, not the held-out checkpoint. Whole-tensor
dequantisation remains an explicit diagnostic cost, not bounded server admission.

## Source reviewed without claiming hardware validation

Additional selected review covered board selection/SIMD/vendor-command paths,
serving metrics/SSE/arrival handling, config fixture helpers, inspector CLIs,
model-family compatibility wrappers, legacy quantisation forwarding, test-process
helpers and NVIDIA direct-ioctl memory/GPFIFO setup. Thin aliases were inspected
as aliases; their backends are not validated merely because forwarding code is
simple. Diagnostic Gemma4 linkname wrappers are not normal production APIs.

The ioctl review found unresolved ownership risks: mapToCPU leaves its successful
mapping descriptor open; GPFIFO setup does not retain the notifier allocation in
the returned object, and failed token acquisition does not release the already
created channel handle. Device handle allocation, mapped-buffer reads/free and
submission teardown need a coherent owner protocol and fake/native tests before
rewriting the experimental ioctl path. No ioctl or GPU diagnostic was executed.
Board vendor subprocess output also lacks a byte bound, and its benchmark JSON
parsing assumptions need validation against the vendor version before its results
are trusted. The earlier GGML/ORT CGo and CUDA capture/global scratch findings
remain open.

## Checks and limits

Each code slice was committed and pushed with targeted tests and whole-tree
vet/build. Logs under `/workspace/tmp/go-pherence-repo-audit/` retain intermediate
failures, including a full sweep that overlapped CLI edits and failed import
compilation. That run is not counted as validation; a fresh sweep uses the
committed tree. That stable sweep exits zero: **101 packages pass and 61 have
no tests**. Vet/build, ARM64/RISC-V builds, 22 Bun tests and the 333-document
link check pass. The attempted independent slot review failed at workspace-path
resolution and adds no review coverage. Remote hash/CI are recorded in the issue update.
ARM64/RISC-V cross-builds are compilation only. Frozen file, binary and record
hashes are checked without reading predictions.

The broader audit is unfinished: unreviewed packages, assembly internals, native
library execution and model-level lifecycle gaps are listed rather than hidden
behind the host test count. The llama.cpp UI port remains pending its hosting
scope decision and did not modify the dirty upstream checkout.

[third]: repository-safety-third-pass-20260919.md
[coverage]: repository-safety-audit-coverage-20260919.md
