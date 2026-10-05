# Pocket TTS fused EOS projection/loss/VJP

## Status

Implemented separate `ptx-resident-eosfused-hybrid-v1` and `vulkan-resident-eosfused-hybrid-v1` scopes. They extend zeroconst execution with one bound EOS affine/BCE/VJP tape, replacing the step's separate forward projection, loss and backward projection tapes. Public standalone FlowLM forward/backward APIs and older scopes retain their existing paths.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. Eight staged inference paths / 469 insertions are unchanged. Both GPU backends were disabled throughout deterministic offline validation. No model, GPU probe, recovery, listening or publication run occurred.

## Contract

The complete-step forward retains transformer hidden audio rows but skips EOS projection only when the explicit fused interface is enabled. One bound tape then computes EOS logits, stable per-position BCE, unweighted compensated mean, weighted logit seeds and affine input/weight/bias VJPs. EOS parameter gradients accumulate exactly once. Only the scalar mean and hidden-input VJP download; logits and logit seeds stay within the tape.

The step-local FlowLM tape retains the returned hidden VJP for backbone backward. Backward consumes it instead of rerunning EOS projection. It rejects missing/wrong-shaped/nonfinite VJPs, a nonzero external EOS seed or a removed/changed fused owner/scope. Other flow condition gradients are added in the existing order. The endpoint input-only flow VJP is unaffected.

EOS objective scaling remains exactly once, in EOS gradient mode 1. Metrics retain the unweighted mean loss. Contiguous-prefix labels and the first padded EOS position use the same helper as the legacy loss builder; inactive hidden gradients are zero. Public fused builders own their inputs/parameters, while private resident construction uses established parameter borrowing.

Before dispatch, admission checks projection topology/finite inputs and a conservative F64 absolute-dot bound with F32 reduction headroom for every row, including padding. Full tape/binding/source-byte admission, pruning/lifetime/zero storage, native ownership and uncertain-failure retention remain unchanged. No new kernel, arithmetic fusion or reassociation was introduced.

## Accounting

Tiny zeroconst versus eosfused complete-step host/mock fixture:

| Measure | zeroconst | eosfused |
| --- | ---: | ---: |
| Tapes | 49 | 47 |
| Boundary tensor downloads | 116 | 113 |
| Commands | 3,958 | 3,959 |
| Conservative logical tape bytes | 144,560 | 144,432 |
| Summed planned owned tape bytes | 34,956 | 34,824 |

The three removed downloads are forward EOS logits, the separate backward projection output and the logit seed boundary. The new tape adds reshapes to bridge affine matrix shapes and EOS vector shapes. Parameter-accumulation counts, resident diagnostics and full snapshot transfers are unchanged. These are offline logical/mock counts, not native peak memory or latency measurements.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-eosfused-20261002`.
Gate window: **2026-10-02 04:37:57–04:38:45 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 170 top-level + 150 subtests |
| Same cases with race detector | 17 top-level + 15 subtests |
| Affected host/mock regressions | 157 top-level + 227 subtests |

Final logs have zero failures/skips. Independent scalar F64 affine/BCE/input/weight/bias VJPs cover weights 0/.1/1/2 and inactive rows. An independently composed legacy projection/loss/projection path matches at existing tolerances. Admission covers masks, nonfinite inputs, weight/budget/topology/overflow bounds and public ownership. Step-local tests reject VJP/seed/scope/owner injection.

New-scope full-step objective endpoints, padding, clipped multistep parameters/M/V/EMA, resume, checkpoint/replay isolation, failures and parameter-accumulation accounting pass. Legacy EOS, projection, FlowLM and CPU/PyTorch full-step regressions remain passing. No numerical tolerance was changed.

Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed, so prior compiler gates were not repeated. Coverage details are in `coverage-functions.txt`; coverage is host/mock only.

### Corrections and review limits

Initial label-helper extraction had a syntax error and was corrected before tests. The fused builder initially assumed the projection output was already a vector; an explicit reshape now provides EOS rank without changing projection arithmetic. Initial download expectations omitted the old backward projection output; the final count correctly records three removed boundaries. Initial and corrected logs are retained.

The self-contained design judge timed out. No independent source approval or native numerical/resource/performance/voice acceptance is claimed. Native BCE/affine behavior, strict-FP/FTZ, released-size resources and sustained quality remain unqualified pending coordinated release of holds.

## Remaining work

Transformer/flow/projection boundaries, graph metadata/nonzero constant construction, dense embedding VJP, diagnostic synchronization and full snapshots remain. This is a further transfer-inclusive hybrid slice, not full-device training or complete-goal acceptance.
