# Pocket TTS fused speaker/sequence tape

## Status

Implemented separate `ptx-resident-voicefused-hybrid-v1` and `vulkan-resident-voicefused-hybrid-v1` scopes. They extend audiofused with raw speaker projection inside prefix/text/audio sequence assembly and its VJP. Legacy APIs/scopes retain separate speaker projection and sequence tapes.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. Eight staged inference paths / 469 insertions remain unchanged. Both GPUs were disabled for deterministic offline validation. No native GPU, model, recovery, listening or publication run occurred.

## Contract

The fused tape accepts raw voice latents, projected audio and a sequence seed. It builds the same `[bos_before_voice, projected_voice, text, audio]` layout, using existing affine, deterministic embedding, slice/pad and vector primitives. Backward returns raw-voice input VJP and audio VJP, accumulating speaker weight, embedding-table and BOS-before-voice parameter gradients once.

Repeated-token embedding VJP remains deterministic and atomics-free. Empty text retains the explicit zero embedding contribution. Empty voice binds a one-row zero speaker input/seed so the speaker parameter is touched with a zero gradient rather than silently missing from the update. Unused forward/reverse work is pruned according to the existing output/accumulation contract.

Forward zero-seed calls download only sequence output. Backward downloads derivatives only, omitting sequence and projected-voice outputs; the standalone projected voice host array is not constructed in this scope. Seeded non-accumulating calls retain derivatives, and accumulating zero-seed calls retain all three parameter roots.

FlowLM captures the explicit fusion interface for backward and rejects its removal/disabled scope. Existing parameter binding, full source-byte admission, owned public constants/read-only private parameter sources, lifetime/zero sharing and uncertain-failure retention remain enforced. Audio/EOS fusion and objective scaling remain unchanged. No new kernel or arithmetic reassociation was introduced.

## Accounting

Tiny audiofused versus voicefused complete-step host/mock fixture:

| Measure | audiofused | voicefused |
| --- | ---: | ---: |
| Tapes | 45 | 43 |
| Boundary tensor downloads | 109 | 105 |
| Commands | 3,962 | 3,942 |
| Conservative logical tape bytes | 144,424 | 141,880 |
| Summed planned owned tape bytes | 34,664 | 34,072 |

Parameter accumulation, resident diagnostics and full snapshot transfers remain unchanged. Counts are summed offline logical/mock evidence, not native peak memory or latency measurements.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-voicefused-20261002`.
Gate window: **2026-10-02 04:53:30–04:54:18 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 210 top-level + 210 subtests |
| Same cases with race detector | 21 top-level + 21 subtests |
| Affected host/mock regressions | 193 top-level + 265 subtests |

Final logs contain zero failures/skips. Independent scalar F64 speaker forward/input/weight VJP and directly indexed sequence/embedding references cover 0/1/3 voice rows, empty/repeated text and BOS/audio seeds. Public ownership, finite/token/shape/budget/topology admission and protected binding inventories pass. Seeded non-accumulating and zero-seed accumulating cases cover empty voice; all three parameter contributions remain explicit.

Full-step objective endpoints, padding, gradients/update/M/V/EMA, clipped multistep/resume, checkpoint/replay isolation, owner/scope/failure guards and transfer accounting pass. Legacy FlowLM/sequence/projection/audio/EOS and pinned CPU/PyTorch full-step regressions remain passing without changed tolerances.

Coverage: scope predicate/public wrapper 100%, graph builder 95.9%, resident evaluator 88.2%; this is host/mock only. Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed; prior compiler gates were not repeated.

### Corrections and review limits

Initial test fixture mixed a two-wide model with three-wide speaker/audio values; it now uses an explicitly specified three-wide reference topology. Initial download accounting omitted the backward sequence output that no longer crosses the boundary; final expectations record four fewer downloads. Development script signature matching was corrected before final gates. Initial/corrected logs are retained.

The self-contained design judge timed out. No independent source approval or native numerical/resource/performance/voice qualification is claimed. Device behavior, strict-FP/FTZ, released-size resources and sustained quality remain unqualified pending coordinated release of holds.

## Remaining work

Audio/sequence and transformer/flow boundaries, graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronization and full snapshots remain. This is a transfer-inclusive hybrid slice, not full-device training or complete-goal acceptance.
