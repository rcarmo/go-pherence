## Repository audit: mapping lifetime, nonfinite values and planner bounds

An advisor that retains an mmap slice can outlive the file that owns it. Clearing
the owner's pointer does not clear a handle somebody already copied, and a later
madvise call can target an address the OS has reused. The safetensors advisor now
detaches before unmap, waiting for in-flight advice and making retained handles
inert. Raw tensor views are a separate contract: their consumers must still finish
before Close.

This continues the [sixth pass][sixth]. The [coverage matrix][coverage] names
selected sections reviewed, not whole-package clearance. The held-out evaluation
and UI hosting decision remain separate unresolved plan items.

## Advice and bookkeeping share one lock

Touch, Prefetch, Evict, cold-eviction eligibility and Detach now use the same
advisor mutex. Cold eviction no longer acts on an unlocked snapshot that may have
become stale. A recent overlapping range protects old aliases of the same pages;
union accounting avoids counting those aliases twice. Smaller touches do not
discard an already tracked extent, and partial evictions conservatively leave a
larger tracked range hot instead of declaring its untouched suffix cold.

Baseline regressions reproduced both overlapping-byte overcount and eviction of
a recently touched overlap. Injected blocking advice checks that Detach and Touch
wait for the syscall boundary; retained-advisor safetensors tests race advice with
Close. Twenty targeted race repetitions pass. The first test run used nonexistent
fixture helper names; that compile error was corrected before validation and is
not counted as a passing run.

Detach does not unmap memory, wait for independent raw readers, or discover other
advisors created for the same mapping. Owners must detach those advisors too.
The reported byte count is the union of conservative advice state, not measured
RSS or an enforceable residency budget. Recomputing that union sorts tracked
ranges, so this fix adds bookkeeping work; no performance improvement is claimed.

## FP16 NaNs and the retained SIMD path

The shared float32-to-FP16 converter previously sent every NaN through its overflow
branch and returned infinity. It now preserves NaN classification and sign,
retains the upper payload bits where representable, and sets the quiet bit. Finite
rounding is unchanged, including its legacy ties-away policy. All 2,046 half NaN
encodings, small float32 NaN payloads, signed zero, infinity, finite ties and
subnormal cases are covered. A narrow independent code review agreed with the
patch and clarified that quieting/truncation is not exact payload preservation.

The whole-tree sweep then found an actual compatibility failure: the retained
AVX2 GELU implementation still narrowed NaNs as infinity. SIMD now stops before
the first NaN-containing vector, leaving that vector and the remainder to the
scalar path. Scanning before writes preserves in-place inputs. The existing
exhaustive half-input plus random-bit test passes bit-for-bit with native dispatch
and with CPU features disabled; no tolerance was relaxed.

That pre-scan has a cost. Three short synthetic samples on 1,269,760 elements
ranged from 0.91--1.32ms before to 1.31--1.37ms after, with zero allocations in
both cases. These noisy, overlapping runs were collected alongside host tests,
not as isolated throughput measurements. They support neither a speedup claim
nor extrapolation to model latency.

## Before indexing native configuration arrays

The LLaMA GGML wrapper now shares one Config type between native and unsupported
builds. Its host-testable validator rejects out-of-range scalar dimensions,
short dtype/override arrays, invalid GQA/RoPE controls, inconsistent per-layer
projection widths and overflowing tensor extents before the constructor indexes
arrays or narrows values to C integers. Optional stub fields and methods now
match the native surface.

Host tests and ARM64/RISC-V test compilation pass. The explicit native tagged
build fails because `ggml.h` is unavailable, so native allocation rollback, dtype
support/block alignment, weight upload bounds, calls after Close and concurrent
decode/Close remain open. Source inspection also found that native MTP setters
are placeholders and a plain GGML context is not retained alongside its allocated
buffer. The constructor guard is not a native lifecycle clearance.

## Sparse rows and bounded planning loops

TRELLIS.2 sparse-tensor validation accepted wrapped coordinate/feature products,
including enormous dimensions paired with empty slices. Checked products now
precede indexing or allocation; direct Coord/FeatureRow access validates the
requested backing extent even when the public struct was forged. Projection
weights and output counts are checked before the SIMD call. Existing numerical
fixtures and new malformed-input tests pass on the CPU; this does not implement
or validate a full TRELLIS pipeline.

Qwen's prefix-key helper could fail to advance for a zero chunk size. Invalid
sizes now produce a zero key and Store rejects them; positive chunk sizes use
remaining-token bounds so offset addition cannot wrap. Layer-window planning
iterates the supplied inventory rather than every integer in an arbitrary window,
and byte accumulation/free-memory conversion saturate instead of wrapping.
Tests use synthetic stats and token IDs, not the frozen evaluation or GPU state.

The Qwen prompt sidecar and older `runtime/kv.ChunkCache` need further budget work:
cloning precedes admission, token/key/metadata and duplicate sidecar storage are
not all charged, and lifecycle is externally serialised. The old chunk cache's
zero budget means unlimited, unlike some newer admission APIs. Those semantics
were recorded, not silently changed in this pass.

## Frozen tokenizer review is source-only

The tokenizer and sidecar files are in the freeze manifest and were not edited.
Selected review covered loading, special-token matching, merge allocation and
fallback behaviour. Whole-file input and repeated BPE merging have no aggregate
work budget; unknown symbols/IDs can be silently omitted. Base vocabulary and
added-token collisions are less strictly checked than sidecar entries, and
unsupported normalisers fall back rather than producing a checked error.
A direct whole-piece vocabulary shortcut also bypasses merge-rank work. These
are open malformed-input/semantic-parity questions, not an authorisation to alter
the frozen prompt/token contract or rescore predictions.

## Checks and remaining plan items

The fresh committed-code NVIDIA-disabled host race sweep exits zero: **106
packages pass and 58 have no tests**. The first seventh sweep failed on the GELU
mismatch and on a TRELLIS import build that overlapped source editing; it is
retained separately and is not substituted for the stable rerun. Targeted races,
whole-tree vet/build, ARM64/RISC-V builds, 22 Bun tests and the 336-document
link/layout check pass. Selected-source coverage is **100/164 packages**, leaving
**64 inventory/tests-only**. Foreign tests were compiled, not executed. Local logs and
before/fixed reproducers are under `/workspace/tmp/go-pherence-repo-audit/`.

Both preserved executables, all 16 freeze-manifest entries and all 676 result
hashes match without inspecting predictions. No completed row was replayed and
no calibration, GPU recovery or service state was changed. The final 1,440-row
results cannot be verified while the run is blocked at 676, and the reusable
llama.cpp UI still needs its standalone-Bun versus Go-embedded hosting decision.
The broader source audit and native ownership findings remain open as well.

[sixth]: repository-safety-sixth-pass-20260920.md
[coverage]: repository-safety-audit-coverage-20260919.md
