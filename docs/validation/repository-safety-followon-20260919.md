## Repository audit: follow-on findings

The first host sweep passed while its shared floating-point comparator silently
accepted NaNs. The next source-review pass exposed that false positive, alongside
request-admission, worker-teardown, mapped-file and graph-lifetime defects. These
are separate findings; none proves the cause of the RTX 3060 bus loss.

This follows the [first-pass audit][first] and updates its [package coverage][coverage].
The frozen evaluation is still stopped at 676/1,440 records. No held-out scoring,
GPU recovery, new weights or hardware restart was used for this work.

## HTTP boundaries

Both LLM and DiffusionGemma servers now require one JSON value under a 1MiB limit,
including trailing whitespace. A LimitReader plus one Decode previously accepted
a valid JSON prefix without proving the body had ended. Oversize returns 413;
trailing/malformed input returns 400. The LLM server retains unknown-field
rejection; DiffusionGemma keeps its existing compatibility behaviour.

DiffusionGemma admits one active generation request, with no waiting queue;
concurrent requests receive 429 before parsing/tokenisation. It checks prompt IDs,
8,192 prompt tokens, 4,096 new tokens, 256 canvas positions, 256 denoising steps
and an aggregate 1,048,576 canvas-position-step budget, including nested config
and checkpoint defaults. These are admission ceilings, not a guarantee that every
maximum fits memory. Cancellation is checked before inference and after each
completed denoising step. Streaming no longer retains an unused duplicate of all
step snapshots. Header/body/idle timeouts are set.

Asset-free mock tests cover oversized/chunked-style bodies, trailing data,
invalid work, busy/retry recovery, streaming and cancellation. An already-running
denoiser/native call is not preempted. Transport output deadlines, authentication,
the LLM inference queue and model-level cooperative cancellation remain separate
review work; this is not production endpoint clearance.

## Workers must finish before memory is unmapped

IME channel and condition pools now serialise Run/Close and join worker goroutines
before Close returns. Repeated Close and Run-after-Close are safe no-ops. The
condition pool can no longer overwrite another submitter's shared fn/phase/done
state. The host lifecycle tests run actual dispatch protocols without affinity
setup or IME kernels: 24 simultaneous submitters, a blocked callback during Close,
zero-worker pools and recovery checks, repeated under the race detector.

The AICPU spin-pool candidate uses the same drain/join ownership before TCM
unmapping, clears its borrowed TCM slices and caps workers at eight distinct
blocks. Its real registration/TCM lifecycle test is platform-gated and cross-built
only; K3 execution remains outstanding. Workers deliberately exit while OS-thread
locked so modified affinity/AI registration does not return to Go's general pool.
Callbacks must return normally and must not re-enter their own pool's Run/Close.

## Copied tensors and borrowed mappings are different contracts

Safetensors copying getters now hold a read lock throughout conversion; Close
cannot unmap midway through GetFloat32/GetInt32/GetBF16. Prefetch serialises its
advisor bookkeeping and coordinates with Close. Nil/repeated Close is safe;
unmap errors preserve the mapping for retry. Converted arrays and shapes are
owned, so callers cannot mutate metadata through a copied getter's shape.

The first version also copied GetRaw shapes. The full suite caught that regression
in OmniVoice's zero-allocation loaders/generation. GetRaw now keeps both bytes and
shape metadata as **read-only borrowed views**, and a regression locks its zero
allocation contract. It is not safe to dereference those views after Close or
race a raw consumer against Close. Model owners must still join inference first.
FrozenGPUEncoder retains its source; Ideogram FP8 linears and DiffusionGemma raw
weight handles depend on source-owner lifetime. No invalid raw pointer was
intentionally dereferenced during testing.

## Numerical tests must reject NaNs

`internal/floatcmp.Close` previously accepted NaN differences because
`abs(NaN) > tolerance` is false. It now rejects NaNs and nonfinite/negative
tolerances, permits only same-sign infinity equality, and widens operands before
subtracting. Dedicated tests cover each case.

This exposed synthetic layered Gemma4/MTP/session fixtures using epsilon zero
with zero projection vectors: RMSNorm(0) generated NaN. The fixtures now use a
positive epsilon. Two older assertions also assumed a removed Gemma4 BF16 layer
boundary and no RoPE-cache growth. They now assert the current F32 scalar path and
isolate the F32 attention-scale oracle with identity RoPE and explicit flash
opt-out; separate F16-KV flash parity tests remain enabled. Production arithmetic
and tolerances were not changed to make these tests pass. Earlier green runs do
not retroactively become numerical validation of those invalid fixtures.

## Attention, convolution and graph planning

The legacy `gpu/` attention helpers are CPU implementations despite their package
name. They accumulated into old output and allocated a score row for every
query/head. They now preflight checked dimensions and use the SIMD runtime with
one score row per call, overwriting output while preserving its tail. Tests compare
an independent float64 scalar oracle, repeated calls, custom/default scales,
vector tails and malformed shapes. Inputs/output must not alias.

For 32 queries, 64 keys, four heads and width 32, the synthetic benchmark changed
from 128 allocations/32,768 bytes to one allocation/256 bytes. Three shared-host
samples were 401--542us before and 130--141us after. This is a narrow helper
measurement, not model throughput or GPU acceleration. Native and CPU-features-
disabled tests pass. Conv1D also rejected stride zero too late (after division);
the shared geometry now checks products/padding and rejects ragged nested tensors
before indexing. Tensor, Whisper and speaker importer tests pass.

The graph planner released the same slot twice for a node consuming `x,x`. Two
later live outputs could then receive the same buffer. A regression fails on the
old planner and passes after unique release. Structural validation now rejects
negative IDs, inconsistent IDs, transient redefinition, invalid/overflowing shapes
and workspace-byte overflow. Added shape metadata is copied. This is not full op
semantic validation: graph mutation after planning, view aliases and persistent
write ownership require an explicit immutable-plan/alias contract.

The placement estimator also counted INT4 packed uint32 words as bytes, dividing
element count by eight without multiplying by four. Layer weight planning now
counts four bytes per packed word and rounds each input row up. Independent
aligned/odd-row formula tests pass. This corrects an estimate, not an actual
allocation failure or a measured recovered-GPU fit.

## Remaining work is explicit

The additional source inspection covers checked arithmetic, float comparison,
half conversion, GGML FP16 lookup dispatch, resource-budget leases, prompt-cache
identity/LRU, graph planning and selected placement helpers. Resource budgets do
not revoke live leases on timeout; callers must release only after work drains.
Prompt-cache budgets currently account snapshot bytes, not token/identity/map
metadata, and clone callbacks execute under the cache mutex on lookup. The cache
needs a clear metadata/admission budget and non-reentrant callback contract.

The core unresolved hardware-sensitive findings are process-global CUDA capture
ownership, returned scratch lifetime, same-model Generate/Close, and K3 execution
of the teardown candidate. Many packages still have only inventory/test coverage;
the matrix does not equate newly read sections with whole-package review.

Raw before/after regressions and failed intermediate sweeps remain under
`/workspace/tmp/go-pherence-repo-audit/`. Each source slice was committed and pushed
with targeted tests and whole-tree vet/build. The final NVIDIA-disabled race
sweep exits zero: **93 packages pass and 69 have no tests**, including the new
HTTP and float-comparison tests. The later placement correction has its own
passing race test and whole-tree vet/build. ARM64/RISC-V builds and K3 lifecycle
test binaries cross-compile; no native K3/GPU execution is claimed. Documentation
checks cover 331 Markdown files with no broken local links. CI/remote hashes are
recorded in the issue update; earlier allocation and fixture failures are retained.

[first]: repository-safety-audit-20260919.md
[coverage]: repository-safety-audit-coverage-20260919.md
