# Pocket TTS computed full-step fixture join

The test-only computed join passes the pinned PyTorch full-step fixture for all 59 parameter gradients, five metrics and final input/logvar derivatives. It executes 12 dependent phases with private candidate arrays. No production source, fixture, backend, CLI or optimiser path changed.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Scope: `/workspace/exports/pocket-fullstep-boundary-proof-plan-review-20261002.md`. Sources:

- `model/pockettts/training_computed_join_test.go`
- `model/pockettts/training_computed_join_negatives_test.go`

## Computation and reference inputs

The unchanged schema-1 fixture is `model/pockettts/testdata/training_step_pytorch.json`, upstream `0acce6b2f390150267557770d2098c5caa9a18ac`, seed 1234. Geometry: frames 3, voice frames 2, hidden 4, latent 2, vocabulary 6, tokens `[1,3]`, mask `[true,true,false]`. JSON config is checked explicitly: `p_equal=.75`, `eos_loss_weight=.1`, `flow_batch_multiplier=1`.

`pocketJoinExecution` contains owned parameters/fixed buffers, batch arrays, noise, times, mask and config. It has no field for expected logvars, derivatives, metrics, gradients or optimiser outputs. Model construction uses the existing fixture helper with only source state and geometry populated. `computePocketJoin` never calls `PocketTrainingStep` or its gradient-map helper.

| Execution output | Producer | Consumer |
|---|---|---|
| Diagonal/distillation logvars | Public weighting tape, zero seed | Early row losses; forward distribution comparison only |
| Audio hidden rows | Public backbone tape, zero seed | Flow conditions, EOS admission, final later graph |
| Unweighted EOS mean | Public EOS projection tape | Reduction total loss; forward projection G is discarded |
| Diagonal input/desired/prediction | Public diagonal forward tape | Public diagonal backward and decoder |
| Distillation primary input, velocity, tangent, endpoint input, desired, prediction | Public distillation forward tape | Public composed endpoint/primary backward and decoder |
| Raw row statistics/logvar seeds and condition/target derivatives | Computed row backward tapes | Reduction/weight/EOS/backbone graph |
| Metrics, scaled logvar derivatives, audio/voice derivatives, direct-tail witness | Real selected-result graph and decoder | Final comparisons only |
| Named parameter G | Row primary/diagonal roots plus disjoint later roots | Final comparisons only |

Expected metrics, forward logvars, derivatives and G are read by comparison code. The isolation test poisons those arrays and also mutates the comparison object's P/batch/noise/mask after freezing execution. A fresh execution still agrees with the unchanged fixture. The main test verifies preservation of all execution P/buffers/batch/samples/config.

The exact manifest is constructed independently of fixture result values. It admits flow 34 names/386 elements; FlowLM/EOS 19/195; weighting 6/25; total 59/606. Four fixed buffers are excluded from all G inventories. Every phase checks exact names/extents, unique parameter/gradient roots and declared gradient occurrences before dispatch. The later graph checks ordered semantic result roles, finite/disjoint owned boundaries, exact inactive logvar derivative zeros and the existing direct-tail copy equality.

## Contribution order

1. Row 0 diagonal: seed `.375` inside the graph; candidate accumulation scale 1.
2. Row 0 distillation: raw primary G at `.125`; endpoint G excluded.
3. Row 1 diagonal: seed `.375`; candidate accumulation scale 1.
4. Row 1 distillation: raw primary G at `.125`; endpoint G excluded.
5. Later disjoint 25 names/220 elements: unit accumulation. Reduction scales raw logvar seeds `.375/.125` once; EOS VJP uses `.1` once; final direct-target derivative is added once.

Forward dummy gradients, EOS forward projection gradients and endpoint parameter gradients never enter the candidates. The total is initialised once. Any stage error returns a nil result; no local candidate map escapes.

## Numerical results

Focused final window: 2026-10-03 12:14:43–12:15:20 UTC. Absolute gates remain unchanged. Counts below describe one full-step comparison.

| Quantity | Values | Mean absolute error | RMS | Maximum absolute error | Gate |
|---|---:|---:|---:|---:|---:|
| Metrics | 5 | 1.19383913e-8 | 2.66560361e-8 | 5.96046448e-8 | 3e-5 |
| Final logvar/audio/voice derivatives | 14 | 8.56977879e-9 | 1.49016335e-8 | 2.98023224e-8 | 5e-5 |
| Flow G | 386 | 7.65099694e-9 | 2.39840198e-8 | 2.16066837e-7 | 8e-5 |
| FlowLM/EOS G | 195 | 2.83953691e-9 | 7.67634009e-9 | 5.12227416e-8 | 8e-5 |
| Weighting G | 25 | 1.20955519e-9 | 5.96071743e-9 | 2.98023224e-8 | 8e-5 |
| Valid forward logvars, diagnostic only | 4 | 0 | 0 | 0 | No new gate |

Existing tighter component tolerances are unchanged. The complete-step fixture supplies aggregate results, with no independent same-geometry row hidden/VJP/statistic/endpoint decomposition. Noise/time adjoints are decoded for finite/ownership checks; the full-step fixture has no corresponding expected arrays. Other complete-step objective configurations, repeated tokens, empty voice and active-last-row independent totals require separate reference work.

## Focused gates

Command pattern, with both GPU backends disabled and no network/toolchain fetch:

```sh
export GOENV=off GOFLAGS= GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off
export GOMAXPROCS=2 GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_DISABLE_VULKAN=1
go test ./model/pockettts -run '^TestPocketComputed(FullStepJoin|Join)' \
  -p=1 -parallel=1 -count=10 -json -timeout=45s
go test ./model/pockettts -run '^TestPocketComputed(FullStepJoin|Join)' \
  -p=1 -parallel=1 -count=1 -race -json -timeout=60s
go vet ./model/pockettts
```

Repeated: 60 top-level + 210 subtest passes. Race: 6 + 21. Zero failures/skips. Formatting and `git diff --check` pass. The first join and first negative-suite attempts also passed; their logs are preserved. No completed component, later-phase or broad gates were repeated.

Rejection checks cover missing/foreign/extent/nonfinite contributions; invalid scale; total extent; sum overflow; duplicate family name; missing/foreign/duplicate/undeclared/misassigned parameter or gradient roots; duplicate/missing selected roots; result count/extent/nonfinite values. Rejected contribution preflight leaves private totals unchanged. Row-1 distillation boundary shape, nonfinite and alias faults discard three prior private contributions. Cancellation at the final graph discards all four row contributions and dispatches no final command. None of these paths publishes metrics/G or invokes an optimiser/checkpoint.

Source comparisons match the accepted three-file interpreter adapter, four-file local coordinator and seven-file generic runtime/test snapshots. Staged inference remains eight files/469 insertions. Fixture SHA-256 `2774178aba94bfcfea47932d557c53c5aaf2a43ba32a12fbe1d7a463070d22ed` is recorded only for provenance; generated numerical results use the gates above.

## Resource ledger

Every complete source is admitted before execution. No pruning, decomposition, ceiling increase or admission bypass was added. Adapter limits remain 4096 tensors/commands/outputs and 1MiB each for its established source/payload/metadata contracts.

Bytes below are explicit F32 payload accounting and the adapter's conservative metadata charge. `Held` includes frozen execution/model state, candidate flow totals, live logvars/hidden/reduction arrays, plus forward row arrays needed by backward. It excludes the comparison object's storage and allocator/GC/RSS overhead. The adapter returns owned copies; all forward dummy G are charged even when subsequently discarded. Per-operation tuple counts are saved in `excerpt.log` and `counts-ledger.json`. There are no groups or persistent commits in this slice.

| Phase | Tensors/commands | Logical source | Source constants | Held | Adapter metadata | Adapter frozen/slots/returned |
|---|---:|---:|---:|---:|---:|---:|
| Weight forward | 53/34 | 2,508 | 868 | 6,639 | 36,071 | 868/824/172 |
| Backbone forward | 269/188 | 29,456 | 6,304 | 6,719 | 187,001 | 6,304/8,900/1,056 |
| EOS forward | 16/9 | 276 | 144 | 6,639 | 10,538 | 144/252/72 |
| Diagonal forward, each row | 532/433 | 12,136 | 2,404 | 6,855 | 391,218 | 2,404/4,568/1,568 |
| Diagonal backward, each row | 532/434 | 12,136 | 2,392 | 6,879 | 391,548 | 2,392/4,576/1,580 |
| Distillation forward, each row | 1,058/864 | 24,224 | 4,776 | 6,855 | 778,712 | 4,776/7,048/1,592 |
| Distillation backward, each row | 1,098/889 | 24,480 | 4,868 | 6,903 | 805,438 | 4,868/6,352/1,596 |
| Later combined graph | 496/339 | 35,676 | 7,816 | 6,855 | 341,483 | 7,816/10,332/1,044 |

The inventory-check arrays add 100/760/20/1,544/880 bytes for weighting/backbone/EOS/flow/later respectively. A selected phase's held arrays + source constants + inventory arrays + adapter frozen/slot/returned payload has a maximum subtotal of 34,743 bytes at the later graph. This subtotal excludes builder temporary graphs, Go maps/slices, construction-time copies and dead arrays awaiting GC. It is not a measured peak or a whole-test upper bound.

The comparison object retains 9,911 bytes of typed-array payload (the JSON ledger charges its three bool mask entries as four-byte values, totalling 9,920). It includes unused AdamW/EMA expectations. The main preservation check holds an additional independent execution snapshot: 2,567 bytes (614 F32 P/buffer values, 25 F32 batch/sample values, two uint32 tokens and three bools). Both fixture reads temporarily retain raw JSON bytes and decoding work, outside graph-phase accounting. Weight `Decode` briefly adds another 172-byte owned result copy while the adapter return is live; hidden extraction adds 48 bytes. Final assembly copies 880 bytes of later G and 24 bytes of diagnostic logvars while the final adapter return is live. No RSS, cumulative allocation, GC liveness or native peak was measured.

Logical source charges summed over the 12 sequential phases are 213,868 bytes; 5,810 commands dispatch. The largest single adapter metadata charge is 805,438 bytes. Summed source charges describe repeated work, not simultaneous live storage. This test carries no performance or released-size qualification.

## Next scope

The bounded computed-boundary milestone is implemented and locally tested. Independent source review has not run; other agents were not contacted or delegated to, per Rui's instruction.

Complete training-engine acceptance still requires integrated update/checkpoint correctness, released-size working-set admission, native adapters/CLI qualification, actual CPU/Vulkan/PTX execution, transfer-inclusive profiles and sustained reliability. The 2,866,551,644-byte legacy resident state exceeds the 2GiB cap before tapes; the 1,839,502,424-byte reduced layout has only generic/mock qualification. This fixture join grants no production/native/optimiser/checkpoint or grouped multi-phase integration permission. Existing execution/publication holds remain unchanged.
