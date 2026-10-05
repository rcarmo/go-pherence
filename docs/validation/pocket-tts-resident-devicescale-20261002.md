# Pocket TTS resident clipping-scale handoff

## Status

Implemented separate `ptx-resident-devicescale-hybrid-v1` and `vulkan-resident-devicescale-hybrid-v1` replay scopes. Validation is offline only, with both backends disabled. HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. The eight staged inference paths / 469 insertions are unchanged. No model, native GPU, listening, recovery or publication run was performed.

This removes the **host clipping-scale data handoff**, not the diagnostic synchronization. Four scalars still download for admission before mutation; CPU dispatch, bias corrections, post-update checks, inter-tape boundaries and coherent full snapshots remain. This is a hybrid, not full-device training or a native performance/quality acceptance.

## Implementation

- `NewDeviceScaleResidentState` requires both global-norm and device-scale driver capabilities before allocation. It owns the same three four-scalar buffers as globalnorm: 48 additional logical bytes, no further resident allocation.
- The coordinator preflights every parameter's six buffer ranges and AdamW constants before norm reduction or any mutation. The host `GradientScale` must equal one. Device contents are not read during preflight.
- Existing globalnorm's maximum scan, fixed-maximum compensated sum and clipping run unchanged. The four-scalar diagnostic result validates finite maximum/scaled sum/scale, zero invalid flag and range bounds before the first AdamW update.
- Separate PTX and Vulkan kernels read `clip[3]` directly. They replace the host scale; they do not apply it twice. The CUDA schedule uses explicit RN multiply/add/divide/sqrt; Vulkan retains precise intermediates. Bias corrections and optional EMA arithmetic follow the existing schedule.
- All six buffers are pairwise disjoint, including gradient and clipping diagnostics. PTX poisons attempted launch/synchronization failures. Vulkan retains failed submitted plans. The coordinator retains uncertain owners and blocks retry, snapshot and ordinary close; successful step accounting waits for the existing mutable-state checks.
- Legacy constructors, replay identities and arithmetic remain unchanged. CLI admission, scope propagation, replay validation, option identity and notices accept the new explicit scopes. CPU defaults remain SIMD.

## Evidence

Directory: `/workspace/exports/pocket-gpu-training-devicescale-20261002`.
Gate window: **2026-10-02 01:52:44–01:55:23 UTC**.

| Gate | Passed |
| --- | ---: |
| New targeted repeated cases (`-count=10`) | 160 top-level + 210 subtests |
| Same targeted race cases | 16 top-level + 21 subtests |
| Boundary/owner follow-up (`-count=10`) | 20 top-level |
| Follow-up race | 2 top-level |
| Final host/mock regressions | 144 top-level + 150 subtests |

Final logs have zero failures/skips. The targeted cases cover independent CPU/F64 norm and clipping references, clipped multistep parameters/M/V/EMA, EMA enabled/disabled, zero/unclipped gradients, checkpoint/resume, padding and objective endpoints, scope rejection, capability/budget admission, first/second parameter preflight, invalid diagnostics, partial mutation and post-update failure retention. Backend mocks check ABI, six-buffer aliases, clipping extent, unit host scale, cancellation, missing kernels/owners, attempted launch/sync/submit failure and blocked retry. Existing numerical tolerances were not changed.

New constructor coverage is 100%; new adapter function coverage is 75–100% in the targeted coverage run. This is function coverage, not native execution coverage. `coverage-functions.txt` records details.

`go vet` on the five affected packages, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout checks and `git diff --check` passed. Offline `nvcc -ptx -arch=sm_80 --fmad=false`, `ptxas`, `glslangValidator -V -S comp` and `spirv-val` passed. The native PTX module now explicitly loads `training_adamw_scaled`; Vulkan creation/close/rollback ownership includes the new 27th pipeline (six bindings, 40-byte push constants).

### Corrections retained in evidence

Initial test construction used incorrect package/mock names, imports and failure-injection fields; these were corrected before passing gates. Vulkan stages expose tensor bindings rather than the private binding list in the test assertion. The initial regression failed only its stale pipeline-count assertion (21, predating several resident slices). The final assertion expects 27 and additionally checks the new device-scale pipeline contract; the complete regression then passed. Initial/final logs are retained, not overwritten.

Independent delegate review did not complete: one request timed out, another could not open workspace files (`Path outside workspace`). No independent source approval is claimed. Direct schedule/admission inspection and deterministic host/mock tests are not native numerical approval.

## Remaining work and holds

No extra norm-summary download was removed in this slice: there is still one four-scalar gradient diagnostic boundary, plus per-parameter post-update summaries and full host snapshots. Native strict-FP ordering, FTZ/subnormal effects, released-size numerical/resource qualification, latency, sustained operation and voice/listening quality remain held pending Rui and `@default` coordination. No hash or byte-exact generated-output comparison was used as numerical acceptance.

Next implementation work: fused boundary/weighting-input orchestration, avoiding host graph/constants construction before pruning, dense embedding VJP/live allocation reductions, fewer inter-tape transfers and smaller completed-step snapshots. No native speedup or full-device scope is claimed.
