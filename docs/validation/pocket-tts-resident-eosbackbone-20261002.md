# Pocket TTS EOS–backbone backward fusion

## Implemented scope

`ptx-resident-eosbackbone-hybrid-v1` and `vulkan-resident-eosbackbone-hybrid-v1` combine EOS affine/BCE/input VJP with conditioner–transformer backward. The weighted EOS hidden VJP stays in the graph and is added once to the flow condition seed before transformer backward. EOS weight and bias gradients accumulate with the other backbone parameters.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The eight-path staged inference index remains at 469 insertions. Both GPU backends were disabled throughout validation. No native/model, listening, recovery or publication run occurred.

## Step contract

The step needs the unweighted EOS mean before its flow loop. A scalar-only EOS projection/loss tape retains the existing absolute-dot overflow admission, labels and reduction arithmetic. It accumulates no parameters and downloads one scalar. The step captures an owned mask and the objective weight instead of a host EOS hidden VJP.

After the flow loop, backbone backward recomputes EOS projection/BCE seed using the immutable graph-derived hidden rows. A seed hook runs between transformer forward and backward: it combines the caller seed with a padded EOS hidden derivative, then builds the existing transformer and conditioner VJPs. Weighting occurs once in EOS mode 1. All backbone/EOS parameter bindings accumulate once, including zero-weight and empty-voice contributions.

The hidden host rows are used only for the same conservative overflow admission; they are not a graph source. Model and batch inputs must remain immutable across paired calls, as in existing training tapes. The public builder owns constants and the private builder borrows only named parameters. Complete source/inventory/byte admission precedes pruning. No cross-tape cache, retained intermediate or kernel was added.

Standalone EOS/backbone APIs and older scopes retain their policies. The new step path rejects changed owners/capabilities, malformed masks and nonzero external EOS seeds. Scalar and input-VJP rejection after execution is terminal: retry and checkpoint boundaries fail, and ordinary close retains the owner.

## Offline accounting

Tiny complete-step host/mock fixture:

| Measure | Backbone | EOS/backbone |
| --- | ---: | ---: |
| Tapes | 39 | 39 |
| Boundary tensor downloads | 101 | 100 |
| Commands | 3,967 | 3,980 |
| Conservative logical tape bytes | 144,112 | 145,328 |
| Summed planned owned tape bytes | 33,168 | 33,320 |

The EOS hidden-gradient download/re-upload disappears. Forward scalar timing, parameter accumulation counts, diagnostic synchronisation and full snapshots are unchanged. Backward recomputes EOS projection/BCE, so commands and logical storage increase. Native latency, peak memory and full-request performance are unmeasured.

## Verification

Evidence: `/workspace/exports/pocket-gpu-training-eosbackbone-20261002`.
Script: `/workspace/tmp/pocket-eosbackbone-gate.sh`.
Final window: **2026-10-02 05:40:59–05:41:43 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 190 top-level + 440 subtests |
| New tests, race detector | 19 top-level + 44 subtests |
| Affected host/mock regressions | 251 top-level + 373 subtests |

Final logs contain zero failures/skips. Cases cover:

- Independently composed standalone EOS projection plus backbone tapes at 1/2/7 frames, empty/nonempty voice, repeated text, padded masks and weights 0/.1/1/2; every input and parameter VJP matches existing tolerances.
- Independent CPU FlowLM/BCE arithmetic using pinned PyTorch model/sequence/Z fixtures; pinned full-step all-gradient/update/EMA, objective endpoint and padding regressions remain passing.
- Deferred scalar forward without parameter accumulation, exactly-once backward binding counts, clipped multistep P/M/V/EMA, checkpoint/resume and replay isolation for PTX/Vulkan mock scopes.
- Full-source byte ceilings, public ownership, finite/shape/mask/weight/overflow/bias/seed rejection, changed owners, dispatch failure and nonfinite scalar/input-VJP post-execution retention.
- Existing conditioner/backbone/transformer/EOS projection/PyTorch and independent F64 EOS/conditioner reference regressions.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout, formatting and diff checks passed. No native kernels changed; compiler gates were not repeated.

Scoped coverage: EOS/backbone builder, capability and admission 100%; scalar evaluator 90.9%, backward evaluator 91.3%; shared absolute-dot admission 100%, seed-hook transformer graph 99.0%. Defensive malformed/aliased driver-return branches are not all covered; the 95% helper target remains open for the evaluators. Whole-package/native coverage is not established.

### Development corrections and review

The first intermediate refactor could not build until the new scope capability method existed. The AST factory guard then caught the new scope missing from both factory switch cases; both were added, with exact replay-hash assertions. Development logs are retained. No numerical tolerance changed.

The read-only delegate could not access the supplied workspace files (`Path outside workspace`); no independent source approval resulted. Native strict-FP/FTZ, released-size resources, sustained throughput and voice quality have not been accepted. GPU and publication holds are unchanged.

## Remaining work

Transformer/flow handoffs, forward EOS scalar tape, graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronisation, full snapshots and live host-state copies remain. This scope still uses hybrid dispatch and host boundaries. Overall implementation and resource/voice acceptance remain active.
