# Pocket TTS conditioner–transformer tape

## Implemented scope

`ptx-resident-backbone-hybrid-v1` and `vulkan-resident-backbone-hybrid-v1` compose conditioner forward, transformer forward/backward and conditioner backward in one resident-bound graph. The sequence and its VJP connect through tensor IDs. Public assembled sequence output is preserved. Backward downloads only raw voice and normalized-latent VJPs.

Training changes are uncommitted at HEAD `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index still has eight paths and 469 insertions. Both GPUs were disabled throughout offline validation. No model, native GPU, recovery, listening or publication run occurred.

## Arithmetic and ownership

The conditioner and transformer share extracted graph-building helpers with their standalone implementations. Those helpers retain operation order, parameters, shapes, exact attention lengths, RoPE tables, final norm, layer scales and VJP arithmetic. All conditioner and transformer parameter gradients bind and accumulate once. Empty voice supplies an explicit zero speaker contribution; repeated-token accumulation, empty text, one-frame audio and zero final normalized-row VJP are preserved.

The full graph admits logical storage, source constants and resident bindings before pruning. The existing 4,096-tensor/command, 16-layer and byte ceilings apply to the combined graph. A tiny 16-layer fixture has 2,613 tensors and 1,956 commands. Public builders own constants; only named parameter sources are borrowed by the private synchronous builder. No cross-tape cache or storage reuse was added.

Zero-seed non-accumulating calls download sequence and transformed rows only, returning owned zero input-VJP placeholders. Seeded non-accumulating calls provide input VJPs without parameter accumulation. Accumulating calls retain every gradient root even with zero seeds, and download only raw input VJPs. The output guard rejects wrong shapes, nonfinite values, borrowed/overlapping outputs and a nonzero final normalized row. Post-execution rejection makes the backend terminal, blocks retry and checkpoint boundaries, and retains its resident owner.

FlowLM captures the backbone backend at forward and checks owner/capability changes before backward. Legacy conditioner and transformer APIs/scopes remain separate. CLI replay uses `resident-backbone-backend` plus the exact scope and runtime pin.

### CLI routing correction

The earlier factory switch admitted only scopes through `resident-devicescale`, although later scopes were admitted by CLI validation and listed in owner routing. Both PTX and Vulkan switch cases now include sampleweight, lifetime, statecheck, zeroconst, eosfused, audiofused, voicefused, conditioner and backbone. A read-only Go AST test checks those factory cases without native initialisation. Successful native construction has not been exercised.

## Offline accounting

Tiny complete-step host/mock fixture, conditioner versus backbone:

| Measure | Conditioner | Backbone |
| --- | ---: | ---: |
| Tapes | 41 | 39 |
| Boundary tensor downloads | 103 | 101 |
| Commands | 3,942 | 3,967 |
| Conservative logical tape bytes | 141,784 | 144,112 |
| Summed planned owned tape bytes | 33,936 | 33,168 |

The sequence remains a public forward download. The backward sequence-gradient download and redundant backward transformer output download disappear. Parameter-gradient accumulation counts, diagnostic synchronisation and full snapshots are unchanged.

Combined backward reevaluates conditioner forward on device; this adds commands and increases logical bytes. Equal-extent intra-tape planning reduces summed owned storage. These counts measure mock dispatch and conservative logical storage. Native latency, native peak memory and complete-request resource costs are unmeasured.

## Verification

Evidence directory: `/workspace/exports/pocket-gpu-training-backbone-20261002`.
Gate script: `/workspace/tmp/pocket-backbone-gate.sh`.
Final gate window: **2026-10-02 05:30:29–05:31:00 UTC**.

| Gate | Passed |
| --- | ---: |
| New tests, ten repetitions | 230 top-level + 410 subtests |
| New tests, race detector | 23 top-level + 41 subtests |
| Affected host/mock regressions | 236 top-level + 329 subtests |

Final logs have zero failures or skips. Tests include:

- Pinned independent PyTorch FlowLM sequence/Z, raw input VJPs and every conditioner/transformer parameter gradient, with existing tolerances unchanged.
- Independent host transformer implementation plus standalone conditioner references over 1/2/7 audio frames, 0/1/3 voice rows and empty/repeated text. Prior independent F64 conditioner reference tests also run in regressions.
- Exact-context variants 0/1/2/128, optional layer scales and final norm, maximum admitted layer count, full-source byte admission, dead-invalid constants, public ownership and private parameter-only borrowing.
- Zero/seeded forward, input-only endpoint execution, exactly-once zero-seed accumulation, full-step objective endpoints, padding, clipped multistep updates, P/M/V/EMA, checkpoint/resume, scope/replay isolation, changed owners and post-execution rejection.
- Shared-helper standalone conditioner and transformer/PyTorch regressions; CPU paths retain existing tolerances.

Affected-package vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout, formatting and diff checks passed. No kernel changed. Existing native compiler evidence was not repeated.

Scoped coverage: backbone capability/public wrapper 100%, result guard 100%, evaluator 97.0%, builder 90.2%; extracted conditioner forward/backward and transformer graph builders 100%. The builder's defensive missing-gradient, impossible-after-admission materialisation/binding and graph-ceiling branches were not all reached. Its 95% admission-helper target stays open; whole-package/native coverage is not established.

### Development corrections

Initial tests used a nonexistent attention mock, incorrect oracle field names/transformer argument order and a float32 period assignment. These test compilation errors were corrected. The post-execution test originally mutated a caller seed, which the graph correctly copied; it now injects a nonfinite downloaded result. A maximum-layer test initially assumed the 16-layer tiny graph exceeded the ceiling; measured counts show that it fits, so the test now checks admitted maximum layers. The regression selector was corrected to include `TestTrainingTransformerTape.*`. Development logs are retained. No numerical tolerance was widened.

A read-only design delegate timed out after 60 seconds. Independent source approval, native arithmetic/strict-FP/FTZ, released-size resources, sustained performance and voice quality have not been accepted. GPU and publication holds are unchanged.

## Remaining boundaries

Transformer/flow and EOS/backbone handoffs, graph metadata/nonzero constant construction, dense embedding VJP, diagnostic synchronisation, full snapshots and live host-state copies remain. The new scope is still hybrid training. Overall implementation and resource/voice acceptance remain active.
