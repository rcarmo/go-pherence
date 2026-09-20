## Repository audit: cache, graph and backend boundaries

The [fourth pass](repository-safety-fourth-pass-20260919.md) addresses remaining LFM2 size formulas, scoped expert-stream ownership and mapping budgets, benchmark completion and inspection error handling. Earlier open-item dispositions below describe this third-pass snapshot.

A cache can respect its snapshot-byte budget while retaining unlimited metadata.
A graph can validate every tensor size while reusing a buffer behind a live view.
Both gaps were present in the next source-review slice, which also expanded
coverage into model metadata helpers and optional native backends.

This follows the [first pass][first] and [follow-on findings][followon]. The
[coverage matrix][coverage] records selected inspected sections, not whole-package
clearance. The frozen evaluation is still blocked at 676/1,440; no held-out
predictions, new weights, GPU recovery or hardware restart were used.

## Retained cache bytes and callback ownership

Prompt-cache admission now charges copied identity bytes, token bytes and a
256-byte per-entry allowance as well as the snapshot payload. Even a zero-byte
snapshot consumes budget, bounding entry count. The allowance is conservative
logical accounting, not a measurement or hard cap on Go RSS/map/allocator overhead.
Snapshot implementations must report retained bytes honestly and return
independent, immutable, concurrently cloneable snapshots.

Declared over-budget snapshots are rejected before Clone; the clone's size is
checked again. Lookup takes a reference to the immutable snapshot under the lock,
then clones outside the cache mutex, so callbacks can inspect cache statistics
without deadlock. Collision-bucket removal clears its tail reference rather than
retaining an evicted snapshot through slice capacity. Tests cover zero-size entry
pressure, pre-clone rejection, callback lock re-entry and eviction reference
release. Retained metadata changes the meaning of UsedBytes: it is no longer only
snapshot payload, and existing budget/eviction fixtures were updated accordingly.

## HTTP admission applies to the LLM server too

The LLM server now admits one generation request and rejects excess requests with
429 before body decoding/tokenisation. It does not retain an unbounded mutex queue.
Model switching uses a separate inference lock; a busy model owner also fails
fast. The HTTP ceiling is 4,096 generated tokens and 8,192 prepared prompt tokens;
zero max_tokens still selects the existing 4,096 default. Header/body/idle
transport timeouts complement the single-document 1MiB JSON decoder.

Cancellation is checked before session creation/prefill and before/after legacy
monolithic generation. Monolithic generation/native kernels cannot be preempted;
a cancelled request can still occupy the admitted slot until that call returns.
Write deadlines, authentication and finer-grained model cancellation remain
separate work. Tests exercise busy/cancel/oversize admission without loading a
model. Existing serial-generation and session tests remain enabled.

## Graph views retain their backing buffers

View, reshape, slice, transpose and contiguous nodes can be zero-copy in a
lowering. The planner now walks backwards through those nodes and extends input
lifetimes through descendant consumers; terminal views retain backing storage
until the end of the plan. This is conservative when a backend copies instead.
Unique releases remain deterministic. New tests cover each view-like op, chains
and returned views, alongside the earlier duplicate-input release regression.
Graphs and nested attribute values must remain immutable while plans exist. This
fix does not validate every op's stride/alias semantics or make graph mutation
thread-safe.

## Shared model helpers now have direct tests

The common GEMV helper could slice with a negative output dimension. Its preflight
now rejects that before slicing. Matrix inspection rejects invalid dimensions;
case-insensitive marker matching now handles mixed-case markers and ignores empty
markers instead of treating them as evidence of tensor coverage. Readiness tests
exercise the runtime/parity/coverage gates; shape summaries own their copied
metadata. Config predicates now have direct tests rather than only importer
coverage.

The weights resolver used Stat rather than Lstat, allowing a dangling index
symlink to appear absent and select a single-file fallback. It now fails closed
on that index. The MOSS audio/model close path clears borrowed adaptor aliases
and model references after successful teardown, so later EnableGPU/EncodeAudio
calls cannot use a closed source. This is sequential post-close protection;
concurrent inference/Close still requires external ownership.

LFM2 metadata inspection now rejects wrapped prompt-plus-generation counts,
overflowing tensor products and invalid GQA grouping. The model remains a
metadata/planning scaffold, not runnable inference. Other layout sizing formulas
still multiply unchecked values and need a broader checked-arithmetic pass;
this slice does not claim to have hardened every LFM2 plan constructor.

## Native slice extents and platform limits

IME packed GEMM previously handed raw pointers to tile loops without checking
full matrix slice lengths. Shared preflight now checks products and complete
A/B/C extents before serial, parallel and pooled entry points. Matching RISC-V
and host packing helpers reject negative/overflowing dimensions and short
buffers. Malformed inputs keep the existing panic contract, now before writes
or native access; zero work is a no-op. Raw-pointer-only exports cannot establish
backing capacities and still require trusted callers.

Transient parallel workers now exit with their affinity-modified OS thread locked,
as persistent workers do, rather than leaking a one-core mask into Go's general
thread pool. The inference packing helper also needed 8*K scratch bytes despite
its 4*K documentation: four rows for the broadcast and four for packed output.
It now checks that extent, requested input length, norm destinations and quantiser
tails before writes. K3 execution remains unavailable; host tests exercise
existing portable packing/scalar paths and malformed admission only.

GGML's shared RawBytes helper now rejects partial blocks and integer overflow
instead of truncating or wrapping allocation sizes. Optional GGML CGo and ORT
implementations were source-reviewed but cannot be built here: their required
headers/libraries are absent. Host tests compile unsupported stubs, not these
native implementations. Concrete unresolved risks found there include:

* `backends/ggmlquant`: DequantRow and VecDot do not validate full encoded extents;
  VecDotRows multiplies row extents unchecked and takes element-zero pointers.
  Arbitrary type IDs/callback availability also need validation before calling C.
* `backends/ggmlgraph` and `backends/ggmlcompute`: several dimension/products and
  C.int conversions precede native allocation/copy. Graph Close/Run lack a shared
  lifecycle lock. Bounds, failure cleanup and concurrent teardown need native
  library-backed tests; stub success is not enough.
* `backends/spacemit/ort`: Run1 trusts caller-supplied outputElems when creating
  an unsafe view of the native tensor, without querying the output type/shape.
  The Go input buffer is retained by an OrtValue across C calls and needs an
  explicit pin/copy lifetime strategy. Session Run/Close is unsynchronised and
  the thread-options status is ignored. These are open safety findings, not
  dismissed as unavailable-platform noise.

MLX/FP8/NVFP4/Q4 preflights were inspected for shape/group/packed-byte bounds and
worker scratch ownership. SIMD wrappers remain dtype-specific: approximate FP8
activation quantisation, input finiteness and input/output overlap need explicit
caller contracts. The expert-stream reader validates file checksums and slot
layouts and joins read workers before returning; returned aligned slot views
are borrowed until reuse/Close, not leases. It accepts trusted manifest paths and
has no aggregate OS memory budget. Those scope limits are recorded rather than
silently replaced by expensive copies or speculative native rewrites.

## Validation and remaining scope

Each fix was committed and pushed in a bounded slice with targeted race tests and
whole-tree vet/build. New test files cover previously untested common config,
internal ops/readiness/tensor inspection and GGML size helpers. GPU/K3/CGo execution
is not established by host checks or cross-compilation. Failed exploratory runs
and full-sweep logs remain under `/workspace/tmp/go-pherence-repo-audit/`; the
final whole-tree NVIDIA-disabled race sweep exits zero: **98 packages pass and
64 have no tests**. Vet/build, ARM64/RISC-V whole-tree compile checks, K3 helper
test cross-builds, 22 Bun tests and 332-document link checks pass. The final
report does not count native GGML/ORT or GPU/K3 execution as validated. CI and
remote hashes are recorded in the issue update. Independent review attempts
timed out and add no corroborating coverage.

The unresolved native issues, CUDA capture/global scratch ownership, remaining
model lifetimes and source-review gaps keep [issue #16][issue] open. This is not
completion of the exhaustive audit or an explanation of Xid79.

[first]: repository-safety-audit-20260919.md
[followon]: repository-safety-followon-20260919.md
[coverage]: repository-safety-audit-coverage-20260919.md
[issue]: https://github.com/rcarmo/go-pherence/issues/16
