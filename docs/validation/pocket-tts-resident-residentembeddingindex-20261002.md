## Pocket TTS resident indexed embedding integration

Two opt-in scopes now route embedding table VJPs through the indexed primitive:

- `ptx-resident-embeddingindex-hybrid-v1`
- `vulkan-resident-embeddingindex-hybrid-v1`

They inherit the previously qualified forward-results hybrid's selected output boundaries, device optimiser and state checks. Earlier scopes and public tape builders remain unchanged. Gradients are still dense; the index removes unrelated-token scanning inside the VJP, not dense table storage or accumulation.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths and 469 insertions. All execution used offline CPU/mock qualification with NVIDIA/Vulkan disabled. CUDA/PTX and GLSL/SPIR-V compilation/inspection were offline only. No native GPU, released-model/audio, recovery or publication work occurred.

## Routing and admission

The new scopes extend the existing capability chain through `forwardResultsEnabled`; ownership still requires one resident backend for all graph components. CLI admission requires explicit experimental-device opt-in, a runtime digest and a byte ceiling. Factory registration covers PTX and Vulkan. Replay hashing uses a distinct `resident-embeddingindex-backend` kind and explicit scope/runtime identity; cross-scope restore is rejected.

`runResidentTape` rewrites only for these scopes, before resident allocation or pruning. `indexResidentEmbeddingTape` first admits the complete source tape and binding inventory. It then copies tensor/command header arrays and appends owned occurrence-index constants for mode-one embedding commands. Changed command operand lists are copied; only the second input ID and mode change. Gradient/output IDs, command order, modes/scales of all other operations, parameter roots and requested outputs remain fixed.

All original tensors/constants remain in the full logical graph, even when unused after rewriting. Combined source/binding/graph/byte admission is repeated before root pruning or device allocation. Late expanded-index rejection preserves the source and produces no driver allocation or gradient accumulation. Unchanged shapes, constants, output arrays and operand lists retain their previous read-only borrowing contract. Newly appended indices and rewritten operands are independent between calls/concurrent compositions.

If there is no mode-one embedding command, the admitted tape is returned without cloning. Gather and already-indexed commands are not changed. The primitive preserves sequence-order reduction, empty-token zero gradients, full dense output and exactly-once parameter accumulation at the unchanged scale. All accumulated gradient roots are still retained by pruning; endpoint forward/backward stop-gradient policy is unchanged.

## Scope and costs

The indexed primitive's stable offsets/row list and full content admission are described in [primitive validation](pocket-tts-resident-embeddingindex-20261002.md). Existing public builders continue to emit mode one; only the new resident scope converts their admitted complete source graphs. The index is rebuilt per graph, with no caching or cross-tape device reuse.

Because rewriting retains the original ID tensor for full-source admission, it adds the entire packed index's `4*(vocabulary+1+rows)` logical bytes, rather than replacing old IDs. Pruning may remove unused old IDs. For vocabulary32,768/rows512, the new constant is 133,124 bytes. Primitive-only host index construction measured about 410 KB/two allocations and 105--110 microseconds; integration also clones headers and repeats admission. These are costs, not native performance results.

The tiny complete row recorded:

| Accounting | Forward-results baseline | Indexed resident scope |
| --- | ---: | ---: |
| Tapes | 11 | 11 |
| Commands | 3,709 | 3,709 |
| Downloads | 42 | 42 |
| Executed logical tensor bytes | 134,064 | 134,100 |
| Planned storage bytes | 41,984 | 42,020 |

Gradient download and accumulation counts remain unchanged. A recording driver inspects the actual pruned resident tape: the new scopes reach exactly one mode-two VJP and no mode-one VJP in the fixture, while the old scope still reaches mode one. No complete-row allocation/latency improvement is claimed. Dense output allocation, dense accumulation, global norm and AdamW/EMA remain; native resource/throughput benefits must include index setup/upload and full request transfers.

## Offline evidence

Evidence directory: `/workspace/exports/pocket-gpu-training-residentembeddingindex-20261002`.
Gate script: `/workspace/tmp/pocket-residentembeddingindex-gate.sh`.
Final gate window: **2026-10-02 10:13:15--10:14:23 UTC**.

| Gate | Passed | Skipped |
| --- | ---: | ---: |
| Selected tests, ten repetitions | 320 top-level + 360 subtests | 0 |
| Selected tests, race detector | 32 top-level + 36 subtests | 0 |
| Affected CPU/mock regressions | 727 top-level + 1,216 subtests | 1 |
| Compiled-artifact follow-up | 32 top-level + 36 subtests | 0 |

Zero failures in final gates. The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` remains skipped with `requires separately reviewed fixed-corpus error-characterisation gate`. It is unrun, not a passed test; no restriction changed.

New tests check source immutability and remap IDs/roots, independently owned index/operand arrays, no-op/gather/already-indexed cases, concurrency, malformed dead sources, binding inventory, complete graph/byte ceilings and rejection before allocation/accumulation. The rewrite helper has 96.3% scoped coverage; the index-build error return is defensive after full mode-one source admission, not an exercised invalid-ID bypass. The scope predicate has 100% scoped coverage.

Both new scopes pass pinned PyTorch full-step gradient/update/EMA comparison, independent CPU objective weights and clipped-update checks, padding, all-tape failure injection, selected-output admission/ownership, checkpoint/resume and cross-scope rejection. Repeated and absent tokens, empty text and empty/nonempty voice pass seeded nonaccumulating and zero-seed accumulating conditioner tests. An isolated mock test compares the admitted dense embedding gradient with the legacy interpreted gradient for repeated/absent token cases; missing non-conditioner branches are synthetic zeros for that test only. Its active boundary remains guarded and is never published as a trainer step.

CLI tests cover explicit scope admission, all component factory conditions, unique/pinned options hashes, the exact new hash-kind formula and honest dense-gradient notices. SIMD and attention-v1 identities retain their existing tests and remain unchanged. Numerical thresholds are unchanged; neither generated-output hashes nor byte identity are used as signal acceptance gates. Exact replay-state comparisons cover the replay contract only.

Package coverage is 87.5% training, 12.4% NVIDIA runtime, 35.1% Vulkan, 68.7% Pocket TTS and 14.7% pockettrain under this selector. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. Offline CUDA12.8 sm80/PTX8.7 `--fmad=false` compilation/ptxas, GLSL compilation/spirv-val and conservative reflection pass. The new integration does not change kernels beyond the previously qualified indexed primitive.

## Review limits and next work

The integration invariant judge timed out after 15 seconds. Prior primitive source/invariant judges also timed out. No independent source or native approval is claimed. Development test-generation mistakes and fixture/API errors were corrected and archived; final gates above passed on the corrected source.

Remaining work includes removing dense gradient allocation/accumulation, transformer/flow audio-hidden handoffs and prediction admission, EOS scalar/diagnostic synchronisation, full snapshots/live host copies and authorised native qualification. These scopes remain resident hybrids, not full-device training. Voice/listening, source approval, native resource/performance and publication holds remain unchanged; this integration does not complete the overall goal.
