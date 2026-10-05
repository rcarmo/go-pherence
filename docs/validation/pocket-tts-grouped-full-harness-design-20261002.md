# Full-Pocket grouped test-harness design

The corrected coordinator still executes only vector mode 0 and compact embedding mode 3 mock tapes. Testing complete Pocket backward recomputation requires a broader test interpreter and independent gradient/result oracles. This is a read-only source design; no harness extension, callable runtime API, model execution or native routing was implemented.

Parent `a90aec30567d485784f2e04d1803c543c9f2817f`. `@default` permitted this design after the correction-only slice in `/workspace/exports/pocket-grouped-recompute-prototype-review-20261002.md`. Runtime seals, staged inference and native/model/audio/publication holds remain unchanged.

## Existing interpreter and required operations

`model/pockettts/training_flow_tape_test.go:16–637` already has a test-only `interpretFlowTapeStorageBound` interpreter. It accepts borrowed inputs, validates external inventory, models slot reuse and propagates borrowed storage through reshape families. It overwrites reused destinations by clearing them before reductions, matching the documented native overwrite contract. It calls `testing.T.Fatal` on errors, so it cannot directly drive recoverable fault-injection tests without a separate test-only error-returning adapter.

The complete combined backbone and flow graphs need these operation families beyond the current two-opcode mock:

| Family | Existing interpreter cases | Qualification needed |
|---|---|---|
| Projection and affine VJP | `linear-forward`, `linear-backward` | All modes, bias/reshape source identity, deterministic reduction order |
| Normalisation and modulation | `norm-forward`, `norm-backward`, `mod-forward`, `mod-backward` | Epsilon, column reductions, mixed/tangent paths |
| Activations | `silu-forward`, `silu-backward`, `relu`, `gelu` | Branch/zero derivative conventions, GELU-tanh schedule |
| Sequence and embeddings | `reshape`, `vector`, `embedding` | Vector modes, compact mode 3/index provenance, repeated/unique token order |
| Transformer | `rope`, `attention-forward`, `attention-backward` | Exact key length, causal/window geometry, RoPE origin, last consumer aliases |
| Flow and row objectives | `time-embedding`, `row`, `lsd-loss` | Selected-time JVP/reverse-over-forward, primary-only roots, endpoint stop-gradient, scale once |
| EOS and guards | `eos`, `eos-admission` | Masked BCE reduction, conservative admission, finite witnesses before any G mutation |

Cases appear at lines 89, 96, 140, 171, 202, 228, 239, 256, 274, 326, 382, 393, 397, 410, 429, 456, 475, 532 and 579. A future harness must enumerate actual emitted op/mode pairs from each tiny source graph and reject unsupported combinations before any group allocates. This table is a source inventory, not proof that every emitted mode has been executed.

`buildStepBackboneTrainingTape` composes reduction, weighting and EOS/backbone, admits the complete graph before pruning, and appends sorted parameter-gradient roots (`training_step_backbone_tape.go:29–105`). Reduced result roots come from `buildStepResultsTape`, preserving logvars/metrics/selected input derivatives/tail witness. `compactResidentEmbeddingTape` adds compact roots only after dense source admission, keeping original IDs and public outputs (`training_resident_embedding_compact.go:11–64`). A test harness must capture root selection at the same stage as production; the earlier parent-root positional error demonstrates why extents alone are insufficient.

## Oracle hierarchy

Existing local schema-1 pinned fixtures use upstream revision `0acce6b2f390150267557770d2098c5caa9a18ac`; tests must verify schema and revision before using them.

- `testdata/training_step_pytorch.json`: `TestPocketTrainingStepPyTorchParity` checks full CPU metrics (`3e-5`), logvars/audio/voice gradients (`5e-5`), and every named trainable gradient (`8e-5`), with exact-zero padded logvar rows (`training_step_parity_test.go:49–90`). This supplies genuinely computed backward expectations; the current prototype only consumes its gradients as constant roots for an update test.
- `testdata/flow_head_backward_pytorch.json`: `TestTrainingFlowTapePyTorchParity` checks flow dual/reverse graph results and revision (`training_flow_tape_test.go:652–689`). Extend grouped closures over the same source without replacing these expected values with grouped outputs.
- `testdata/flowlm_training_pytorch.json` and `transformer_backward_pytorch.json`: existing backbone/transformer parity tests provide independent component VJPs (`training_backbone_tape_test.go:12–72`, `training_transformer_tape_test.go:16–73`).
- Existing diagonal/distillation device-input reference tests compare retained-source math to CPU/F64 (`training_diagonal_device_inputs_test.go:434`, `training_distill_device_inputs_test.go:421`). They cover the original ownership/scaling semantics but do not already establish grouped execution.

Keep current tolerance values and nonfinite checks. Report mean/RMS/max errors for grouped versus independent expected gradients/results and clipped updates. Existing ungrouped interpreter comparisons are useful schedule checks, but cannot replace independent PyTorch/F64 expectations. Missing mask/empty-voice or objective-endpoint oracle variants must remain explicitly open; do not download assets or manufacture self-referential expected arrays.

## Suggested bounded test-only extension

1. Use tiny pinned fixture geometry only. Load its constant arrays, construct the existing complete source graph with unchanged public builders and privately copied metadata/values, then validate the full gradient inventory and all op/mode coverage before splitting groups.
2. Extract an error-returning interpreter adapter into a new test-only file, preserving arithmetic/dependency order and borrowed reshape semantics. No package production API is required. The adapter should expose completed owned output buffers and injectable construction/run/download/close boundaries.
3. Keep all P sources separate from selected G roots. Frozen P/seed/mask/constant snapshots apply across every group; a gradient must be tested while reading another group's P. Non-stateful pure commands only; no reads from current G or output buffers reused across groups.
4. Preserve the corrected single-owner fence, pre-copy metadata checks, conservative cumulative plan accounting, cancellation checks after callbacks/successful close and before each constructor/dispatch, and exact terminal retention. Failure after a previous group commits remains terminal even with a nil new owner.
5. Recompute complete result/admission witnesses in every group initially. Compare their content, publish once after all closes, and validate the witness before any G mutation. Test late guard failure after earlier groups without exposing metrics/replay/checkpoint/update.
6. Compare every grouped trainable gradient and selected input/logvar/metric result against the pinned fixture. Then test clipped multistep P/M/V/EMA against the established host schedule and available independent update expectations, with unchanged thresholds.
7. Cover group ordering/boundaries, repeated/unique tokens, partial masks and empty voice where the oracle supports them; objective weights 0 / .25 / 1; primary-only gradients, endpoint stop-gradient and unit parameter accumulation; snapshots/resume remain the existing production contracts, not new routing.

Before execution, assert group union exactly once, root distinctness/order/extents/producer modes, full external binding inventory including dead sources, compact index validity and all plan/copy descriptors. Postexecution actual-buffer validation remains distinct from pure topology admission. Never reduce original logical admission to a group-owned plan.

## Host and planner costs

The current test coordinator copies source constants plus current P before all groups, retains all pruned plan metadata, and copies result arrays. A full-Pocket harness therefore needs a separate host ledger:

`hostFrozenPayload = 4*sum(source Values lengths) + 4*sum(bound P extents) + 4*sum(published result extents)`

`metadata = original shapes/operands/output IDs + group/name maps + all retained pruned shapes/operands/roots/remaps + bounded planner scratch`

The corrected 1 MiB prototype metadata byte charge is a conservative synthetic limit, not measured allocator RSS and not a released harness ceiling. Full-source Values may duplicate resident parameter data across graph constants, so the host freeze can exceed one parameter set even when device borrowing deduplicates it. Graph arrays can share fixture backing before freezing; defensive copies destroy those aliases intentionally. Document both logical slices and unique payload where relevant.

For large sources, planning one group at a time could reduce retained metadata but would complicate all-group preflight and callback mutation guarantees. Reconstructing plans after an immutable summary pass is a design alternative needing its own bounds and validation; it is not implemented. Released YAML resource-planner tests remain separate from small numerical execution fixtures.

## Open acceptance gates

No complete Pocket grouped backward execution currently exists. The current test mock's pinned update proves coordination with supplied gradients only; its F64 compact/vector fixtures do not cover attention/norm/mixed-AD. Real driver pipelines, scratch/staging, completion/FTZ, physical peak, throughput, cancellation soak and voice/listening evidence remain unqualified. Extending a test-only interpreter does not grant native/CLI/runtime approval. Return any proposed test-harness implementation scope to `@default` before changing the submitted coordinator or callable code.
