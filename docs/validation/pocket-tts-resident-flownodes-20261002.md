## Pocket TTS inline flow AD-node inputs

Flow reverse metadata now stores its unary/binary inputs inline. The profile identified separately allocated `graphDual` operand lists at all six node-emission sites. Every flow AD operation has at most two inputs, so a fixed two-slot array removes those allocations without an arena, cache or new numerical operation.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. This slice uses offline CPU/mock qualification with NVIDIA/Vulkan disabled. No native GPU, released-model/audio, recovery or publication work occurred.

## Metadata contract

Private `graphNode.input` changes from `[]graphDual` to `[2]graphDual`. Linear, SiLU, norm and slice nodes populate/read slot zero; modulation and add nodes populate/read both slots. Unary nodes leave slot one zero. All consumers in reverse emission and reverse command-count simulation already use those indexed accesses; none depend on length, nil, append, serialization or shared backing-array mutation.

Inline arrays make a node value copy own its operand slots. Cloning a node-header slice also copies those slots, rather than sharing their old separate backing arrays. Ordinary shallow copies of the node slice itself still share the schedule; this change does not claim otherwise.

Node order, output/parameter/operand IDs, fields, arithmetic, scaling and reverse command-count rules are unchanged. Public tapes contain emitted commands, not private AD nodes. Complete-source/binding admission, root closure, owned public inputs/results, endpoint stop-gradient, gradient accumulation, retention and backend/replay scopes are unchanged. No tolerance or kernel changed.

On amd64 a node grows from 96 to 104 bytes because a 24-byte slice header becomes a 32-byte inline pair. Each unary node avoids a 16-byte operand array and each binary node avoids a 32-byte array. Existing node-slice capacity/growth remains unchanged; the complete-row measurement below includes its larger elements and shows a small net byte reduction. The private schedule is temporary builder storage, not exposed through the returned tape.

Future operations with more than two inputs would need an explicit representation change. Future generic consumers must not interpret the unused unary slot as a real tensor ID: zero is a valid ID, not an arity marker. The current operation switch determines arity.

## Complete-row measurements

Before/after binaries differ only in private node representation and the corresponding synthetic test literal. Both include the previously qualified composition operand storage. Fixture preparation was outside measurement; offline Go, `GOMAXPROCS=2`, `-p=1 -parallel=1` and both GPU disable flags were fixed.

Initial alternating five-sample, 100-row measurements:

| Path | Before median B/op | After median B/op | Before allocations/op | After allocations/op |
| --- | ---: | ---: | ---: | ---: |
| Public tiny row | 1,743,904 | 1,741,076 | 1,735 | 1,579 |
| Private tiny row | 1,910,132 | 1,907,352 | 1,781 | 1,626 |

The short private after median includes rounding/noise in per-row allocation accounting. A longer isolated seven-sample comparison, with 500 warmed private rows per sample and alternating starting order, measured:

| Private row | Before median | After median |
| --- | ---: | ---: |
| Allocated bytes/op | 1,910,116 | 1,907,298 |
| Allocations/op | 1,781 | 1,625 |
| Time ns/op | 1,173,517 | 1,151,702 |
| Time range ns/op | 1,141,825--1,229,543 | 1,105,308--1,213,155 |

This saves **156 allocations/op (8.8%)** and about **2.8 KB/op (0.15%)**. Timing ranges overlap, so no latency gain is established. Initial short private timings had a slower after median (1.163 / 1.229 ms) with overlapping ranges; a first longer comparison overlapped the gate run and is retained as contaminated timing evidence, not used for acceptance. The final isolated comparison had no concurrent gate launched by this session. These tiny mock measurements do not establish native performance.

Separate bounded 20-row `-memprofilerate=1` attribution runs allocated **42,438 -> 38,926 objects**, about 8.3% fewer including setup. Node operand allocation sites disappear from emission profiles. Post-GC total in-use profiles measured **42.96 / 42.85 KiB**, essentially unchanged. Test/regexp setup dominates those totals; they do not prove peak RSS, released-size residency or native allocator behaviour.

## Offline gates

Evidence directory: `/workspace/exports/pocket-gpu-training-flownodes-20261002`.
Gate script: `/workspace/tmp/pocket-flownodes-gate.sh`.
Window: **2026-10-02 09:30:18--09:31:24 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 170 top-level | 0 |
| Selected tests, race detector | 17 top-level | 0 |
| Affected CPU/mock regressions | 633 top-level + 1,178 subtests | 1 |

Zero failures. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

The new test checks both host/device-time graph construction, all six node kinds, active operand ID bounds, unused unary slots, node-value copy isolation and independently cloned schedule isolation. Existing reverse-count tests exercise multiple block counts and host/device time, including bounded oversized synthetic schedules. Existing ownership/cross-call/concurrency, complete-source rejection, CPU/F64/pinned PyTorch mixed-AD/FlowLM/full-step, objective scaling, padding, gradients/updates/EMA, checkpoint/resume, terminal retention and accounting gates pass unchanged tolerances.

All six node emitters, reverse emission and reverse metadata-count/reservation helpers have **100% scoped statement coverage**. The entire flow builder file is not fully covered. Package coverage under the regression selector is 87.0% for `backends/training`, 68.6% for Pocket TTS and 14.4% for pockettrain.

Vet covers all three affected packages. Host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, model-layout and diff checks pass. No native execution or shader/compiler change is involved.

## Review limits and remaining work

An invariant judge agreed that the representation preserves indexed unary/binary semantics and improves node-value ownership, while calling out zero-slot conventions and future length/layout-sensitive consumers. It did not inspect source or execute native code; independent source/native approval is not claimed.

The mock interpreter remains the largest flat allocation contributor, but it is not a production GPU optimisation target. Remaining production graph metadata and full-source/binding admission should not distract from device completeness: transformer/flow audio-hidden handoffs, forward prediction admission, dense embedding VJP, EOS diagnostics and full snapshots remain. Native resources/performance and new-voice/listening acceptance are held. This offline slice releases no hold and does not complete the overall implementation goal.
