## Pocket TTS private LSD diagnostic copies

The private resident training-step LSD call now returns before copying velocity, time derivative and endpoint arrays or materialising a host residual array. It returns raw loss/square statistics and the existing condition/target/logvar gradients. Public calls still return four owned diagnostic arrays. Existing scopes, device graphs, roots, arithmetic and admission are unchanged.

This is a small host-allocation reduction. Full graph construction dominates the complete tiny mock row and native performance is unmeasured.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference is unchanged: eight paths and 469 insertions. NVIDIA/Vulkan were disabled. No native GPU, released-model/audio, recovery or publication run occurred.

## Result and execution contract

Only step-only calls on enabled resident-distillresults descendants use the early return. The forward computes and validates the same primary velocity/tangent, endpoint input/prediction and desired values. The same fully admitted distillation graph then executes with the unchanged primary-only parameter scale, condition/target derivatives, loss statistics and finite noise/time witness. No diagnostic host copy feeds those computations.

Private `Velocity`, `TimeDerivative`, `Endpoint` and `Residual` fields are nil. The training step consumes only `RawSquare`, `Loss` and the selected gradients. Public combined calls and earlier step scopes retain their full diagnostic behaviour. Public arrays remain finite, owned, disjoint from each other, caller inputs and returned gradients.

Failed forward/backward still propagates the existing terminal error and retains owners. The removed operations had no synchronisation, logging, admission or lifetime role. Device downloads, tape counts, parameter roots/scales and replay identities are unchanged.

## Bounded mock allocation samples

`TestTrainingLSDPrivateDiagnosticsAllocation` measures a complete warmed tiny CPU/mock row after preparation, including graph build, admission, pruning, interpretation, downloads and gradient accumulation. The test was run five times before and after with `-benchtime=20x`, `GOMAXPROCS=2`, `-p=1 -parallel=1` and both GPUs disabled. Raw logs are retained.

| Path | Before median B/op | After median B/op | Before median allocs/op | After median allocs/op |
| --- | ---: | ---: | ---: | ---: |
| Public row | 2,687,727 | 2,687,717 | 8,934 | 8,934 |
| Private row | 2,854,036 | 2,853,958 | 9,004 | 9,000 |

Four allocations disappear from the private row. The two-channel fixture's four F32 arrays contain 32 bytes; measured total byte deltas include allocator/graph noise and are too small for a whole-row memory claim. Private row median timing moved from 1.513 ms to 1.604 ms, with overlapping noisy sample ranges; no latency gain is established. Public medians moved from 1.467 ms to 1.404 ms without a code change in that path. The private row still allocates more overall than the public row because its finite-witness graph has additional metadata. This slice does not reduce that cost.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-lsddiagnostics-20261002`.
Gate script: `/workspace/tmp/pocket-lsddiagnostics-gate.sh`.
Final window: **2026-10-02 08:19:55--08:20:55 UTC**.

| Gate | Passed |
| --- | ---: |
| Selected correctness tests, ten repetitions | 70 top-level + 150 subtests |
| Selected correctness tests, race detector | 7 top-level + 15 subtests |
| Affected CPU/mock regressions | 450 top-level + 791 subtests |

Final logs have zero failures/skips. Tests compare public/private statistics and raw condition/target/logvar gradients at parameter scales 0/.25/1 for PTX/Vulkan mock distillresults and forwardresults scopes. After zero non-flow branches, accumulated primary roots match public roots. They check nil private diagnostics, public ownership and unchanged earlier-scope diagnostics. Input/forward/backward rejection returns no diagnostics; uncertain execution prevents boundary validation and ordinary close.

Independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step regressions pass unchanged tolerances, including objective endpoints, padding, clipping, AdamW/M/V/EMA, checkpoint/resume and retention. Existing tiny tape accounting passes unchanged. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass.

All statements in the changed private early-return branch execute under the selected tests. Public/private wrapper coverage is 100%; the larger existing LSD orchestrator is 93.1% in this regression selector, including older unfused error branches. Package coverage is 57.2% for Pocket TTS and 14.4% for pockettrain. No new numerical/admission helper or kernel was added, and these scoped figures do not establish native coverage.

### Review and limits

A mathematical judge found no invariant issue and asked whether the removed copies served synchronisation, error, logging or lifetime functions. Inspection confirms they were owned diagnostic materialisation only; graph execution, validation and error paths remain. The judge did not inspect source or run native code. Independent source approval, native resources/throughput and voice/listening acceptance are open. Numerical thresholds and holds are unchanged.

## Next work

Graph metadata and nonzero constant materialisation still dominate host construction. Audio-hidden transformer/flow handoffs, prediction admission, dense embedding VJP, EOS scalar/diagnostic synchronisation and full snapshots/live host copies still require work. Continue bounded offline implementation before authorised native qualification.
