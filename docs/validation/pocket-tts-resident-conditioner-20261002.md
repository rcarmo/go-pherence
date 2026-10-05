# Pocket TTS fused conditioner tape

## Status

Implemented separate `ptx-resident-conditioner-hybrid-v1` and `vulkan-resident-conditioner-hybrid-v1` scopes. They combine raw speaker/text sequence assembly with shifted-audio/BOS projection and VJPs in one bound conditioner tape. Legacy scopes retain separate audio and speaker/sequence tapes.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. Eight staged inference paths / 469 insertions are unchanged. Both GPU backends were disabled throughout deterministic offline validation. No model, native GPU, recovery, listening or publication run occurred.

## Contract

The conditioner accepts raw voice, normalized audio, token IDs and sequence seeds. It forms the same `[bos_before_voice, projected_voice, text, projected_shifted_audio]` sequence. Speaker projection, repeated-token embedding gather/VJP, shifted-audio projection, BOS gradients and input VJPs stay within one graph. All five parameter gradients accumulate exactly once.

Forward zero-seed calls download sequence only. Backward requests raw-voice VJP when voice exists and normalized-latent VJP; it does not download projected audio or the projected-audio seed. The last normalized-latent row remains zero. Empty voice still contributes an explicit zero speaker gradient; empty text retains a zero embedding contribution.

The step-local FlowLM tape captures the conditioner interface, omits separate projected voice/audio arrays and rejects removed/disabled owners before backward. Seeded non-accumulating calls retain input VJPs; zero-seed accumulation retains all five parameter roots. Result field `DAudio` here denotes normalized-latent VJP, not projected-audio seed, with explicit dimensional admission.

Full source/topology/finite/token/binding/byte admission, public owned constants/private read-only parameter sources, pruning/lifetime/zero allocation and failure retention remain enforced. Prior EOS fusion, objective scaling and standalone legacy paths are unchanged. No new kernel or arithmetic reassociation was introduced.

## Accounting

Tiny voicefused versus conditioner complete-step host/mock fixture:

| Measure | voicefused | conditioner |
| --- | ---: | ---: |
| Tapes | 43 | 41 |
| Boundary tensor downloads | 105 | 103 |
| Commands | 3,942 | 3,942 |
| Conservative logical tape bytes | 141,880 | 141,784 |
| Summed planned owned tape bytes | 34,072 | 33,936 |

Five parameter-gradient accumulation counts, resident diagnostics and full snapshot transfers are unchanged. Counts are offline logical/mock evidence, not native peak memory or latency measurements.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-conditioner-20261002`.
Gate window: **2026-10-02 05:02:01–05:02:45 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 230 top-level + 230 subtests |
| Same cases with race detector | 23 top-level + 23 subtests |
| Affected host/mock regressions | 214 top-level + 288 subtests |

Final logs contain zero failures/skips. Independent scalar F64 references cover the complete sequence and all raw-input/five-parameter VJPs at 1/2/7 audio frames, 0/1/3 voice rows and empty/repeated text. Independently composed audio and speaker/sequence tapes match existing tolerances. Cases cover public ownership, shape/finite/token/bias/budget admission, final-row zero, seeded non-accumulation, empty-voice zero-seed five-binding accumulation, direct capability/scope rejection and owner retention.

New-scope full-step objective endpoints, padding, gradients/update/M/V/EMA, clipped multistep/resume, checkpoint/replay isolation, failures and boundary accounting pass. Legacy FlowLM/sequence/projection/audio/voice/EOS and pinned CPU/PyTorch full-step regressions remain passing with unchanged tolerances.

Coverage: scope/public wrapper 100%, builder 95.5%, resident evaluator 94.6%; this is host/mock coverage. Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed; prior compiler gates were not repeated.

### Corrections and review limits

Initial full-step download expectations included the projected-audio output/seed now removed. Copied owner tests initially required the separate audio/voice fusion fields rather than the combined step-local conditioner; those assertions now check the conditioner field. Development exact-match edits were completed after a signature mismatch. Initial/corrected logs are retained; no numerical gates were relaxed.

The self-contained design judge timed out. No independent source approval or native numerical/resource/performance/voice acceptance is claimed. Device behavior, strict-FP/FTZ, released-size resources and sustained quality remain unqualified pending coordinated release of holds.

## Remaining work

Conditioner/transformer and transformer/flow boundaries, host graph metadata/nonzero constants, dense embedding VJP, diagnostic synchronization and full snapshots remain. This is a transfer-inclusive hybrid slice, not full-device training or complete-goal acceptance.
