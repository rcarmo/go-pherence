## Compact embedding gradient accumulation primitive

An optional driver primitive now accumulates compact embedding VJP slots directly into dense resident ping-pong gradient storage. It avoids a separate vocabulary-sized expansion of compact tape output. The resident coordinator, ordinary binding validator and CLI/replay routes are not changed by this slice; compact graph integration remains follow-up work.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. All execution used offline CPU/mock qualification with NVIDIA/Vulkan disabled. CUDA/PTX and GLSL/SPIR-V compilation/inspection were offline only. No native GPU, released-model/audio, recovery or publication work occurred.

## Formula and lookup

The compact index is the admitted layout from [compact VJP validation](pocket-tts-resident-embeddingcompact-20261002.md): ascending unique active-token slots, trailing -1 padding, sequence-ordered occurrence rows and complete coverage. The new kernel consumes the already-reduced slot values. For every dense output element:

`output = previous + scale * contribution`

Contribution is that token's compact slot/column value if present, otherwise positive zero. A lower-bound binary search treats negative padding as greater than every valid token. Bounds keep the search inside `rows` slots; a successful match reads only `slot*width+column`. No atomics or cross-thread reduction is needed. The scale applies once after the sequence-order VJP sum, not once per occurrence. CUDA uses `--fmad=false`; GLSL marks the scaled term precise to preserve separate multiply/add rather than contract into FMA.

Absent tokens still evaluate `previous+scale*0`, rather than simply copying `previous`; this preserves the existing dense accumulation's signed-zero formula. Finite negative scales are supported by the primitive, like the vector operation. The training coordinator's existing nonnegative scale policy remains unchanged.

The shape contract requires admitted embedding limits (rows1--1024, width1--16384, vocabulary1--65536, vocabulary*width<=2^26) and finite scale. Buffers have exact extents: two dense buffers of vocabulary*width, compact rows*width, and index3*rows+1. Output must not overlap any read binding. Read-only inputs may alias if their shapes/ranges support it; no writable alias is admitted.

## Admission, lifetime and errors

`ResidentCompactEmbeddingDriver` is an optional interface with pure validation and synchronous execution methods. It does not relax existing resident binding extents. Future coordinator integration must validate the complete source/binding/index inventory and preflight every accumulation before submitting any gradient mutation. Raw callers remain responsible for admitted immutable index contents; runtime wrappers validate metadata, not device-buffer contents.

PTX preflight checks context/operator, buffer ownership/liveness, exact extents/output overlap, finite scale and kernel presence under the operator lock. Execution repeats admission, launches the eight-argument kernel and synchronises. Launch/synchronisation failures poison the operator and prevent retry. Borrowed tape buffers remain checked by the existing conversion helper.

Vulkan preflight checks driver/context/terminal state, exact extents, live reshaped views and the stage descriptor/push/alias contract without submitting work. Execution constructs a synchronous single-stage plan. Submit failure retains the plan and marks the driver terminal; Close and retry reject the uncertain boundary. Existing tape/arena owners must remain alive through this operation. No fallback, recovery or buffer ownership transfer was introduced.

## Storage and work limits

Compact VJP output is rows*width rather than vocabulary*width, so it need not be expanded into a dense tape temporary before accumulation. For vocabulary32,768/width1,024/rows512, the compact VJP's theoretical output is2MiB instead of128MiB. That saving depends on later integration selecting compact bindings; current resident scopes still produce dense tape gradients.

The accumulator still reads/writes dense ping-pong resident gradients and performs dense work. Token lookup is logarithmic in padded rows for every dense element; it may cost more arithmetic than the ordinary vector accumulator. Persistent dense parameter/gradient/scratch/optimiser buffers, global norm, AdamW/EMA and snapshots remain unchanged. No native memory, speed or complete-request saving is established.

Device operator construction now requires one additional `training_embedding_accumulate` entry in the shared PTX module and one Vulkan pipeline (factory inventory27->28, four bindings/16 push bytes). Existing scopes do not dispatch it, but construction/module overhead increases even there. Factory ownership/close tests include the new pipeline. Runtime/device identity pins must continue to bind the actual compiled implementation; offline tests do not verify a caller's declared pin.

## Numerical and mock evidence

Independent references expand sequence-ordered token sums and apply the legacy dense accumulation formula. The tested binary-search formula uses only compact slots. Repeated, missing, maximum token, all-active and padded-bucket cases use zero/positive/negative scales and nonzero prior gradients. Across384 output comparisons, absolute F32/F64 error has mean **7.57912251e-11**, RMS **2.10960951e-10** and maximum **1.16415322e-9**. Existing2e-6 gates remain unchanged; this bounded corpus is not broad error characterisation or native FTZ/rounding acceptance.

Signed-zero tests check the actual absent-token arithmetic contract. No hash or generated-output byte-identity gate is used. Formula tests do not execute compiled kernels numerically.

PTX mocks check eight-argument pointer order, shape constants, scale, grid/local size, pure preflight launch count, bad extents/scale/context/owner/alias, missing kernel and terminal launch/sync behavior. Vulkan mocks inspect shader bindings/push/local size, stage constants/groups, bad extents/scale/context/owner/alias, successful plan execution, submit failure, retained plan, rejected Close and retry. The factory test checks the new pipeline's creation position and ownership. Existing indexed/compact VJP and resident regression gates remain passing.

## Offline gates

Evidence directory: `/workspace/exports/pocket-gpu-training-embeddingaccumulate-20261002`.
Gate script: `/workspace/tmp/pocket-embeddingaccumulate-gate.sh`.
Final window: **2026-10-02 10:37:21--10:38:28 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 240 top-level + 50 subtests | 0 |
| Selected tests, race detector | 24 top-level + 5 subtests | 0 |
| Affected CPU/mock regressions | 738 top-level + 1,216 subtests | 1 |
| Compiled-artifact follow-up | 24 top-level + 5 subtests | 0 |
| Numerical follow-up | 10 repetitions | 0 |

Zero failures in final gates. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

Shape validation, PTX raw operation/preflight and Vulkan stage have100% scoped coverage. PTX optional-driver execution has83.3%, Vulkan preflight96.3% and execution81.2%; defensive post-preflight conversion/reshape errors are not all exercised. Package coverage is88.3% training,12.9% NVIDIA runtime,36.7% Vulkan,68.7% Pocket TTS and14.7% pockettrain under this selector. No whole-backend/full-device coverage claim follows.

Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. Offline CUDA12.8 sm80/PTX8.7 compilation uses `--fmad=false`, no fast math, and passes `ptxas`. GLSL passes compilation, `spirv-val` and conservative reflection without widening the opcode envelope. Development fixture/helper API mistakes were corrected and their failed logs retained.

## Review and next work

The invariant judge timed out after15seconds. No independent source or native approval is claimed. Next add explicitly validated compact-gradient binding metadata and coordinator support, retain index/output roots through pruning, preflight the complete accumulation pass and preserve terminal owner retention before a new CLI/replay scope can use it. Dense resident gradient buffers and optimiser work remain.

Transformer/flow handoffs, prediction admission, EOS diagnostics, snapshots and authorised native resources/performance/voice qualification are still open. All execution/recovery/listening/publication holds remain. This primitive does not complete the overall implementation goal.
