## Resident producer lease

`ResidentTapeLease` keeps one completed read-only producer alive across multiple pruned consumers under the same `ResidentState`. Consumers borrow pinned producer roots, use current resident parameters and optionally accumulate their own named gradients. The producer runs once and closes after all successful consumers. This primitive has no CLI or model routing yet.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference is unchanged: eight paths / 469 insertions. Native execution, model/audio/listening and publication holds are unchanged.

## State ownership and admission

`BeginPrunedTapeLease` validates the complete producer source and parameter inventory, resident names/extents, pinned declared outputs, positive logical ceiling and handoff-driver capability before allocating. Backend preflight admits all kernels and existing buffer liveness/overlap. It copies graph metadata/constants, prunes selected roots and records original-to-pruned IDs. `BeginZeroPrunedTapeLease` additionally selects existing lifetime allocation and immutable positive-zero sharing; the constructors protect borrowed constants.

An active lease is separate from the invocation-local `borrowed` flag. Public update, snapshot, host accumulation, ordinary tape execution, a second lease and state close are blocked. The authorised consumer runner checks exact lease/state identity before entering the existing execution/accumulation path. Copies of the lease cannot act as the state's current lease. It sets the invocation borrow before producer-output borrowing or consumer preflight, so callbacks cannot re-enter lease close, downloads or consumer execution.

Each consumer validates its full graph, parameter inventory and declared downloads before pruning. The combined ceiling charges the complete producer and consumer sources, including borrowed placeholders. Edges require a pinned producer output, a constant consumer target, equal extent, no parameter-binding conflict and no duplicate target or producer reshape family. All bound edge buffers then pass backend liveness/extent/overlap and kernel preflight before consumer allocation.

Pruning remaps live edge targets as well as parameter/gradient/compact-index roots. Dead edges remain admitted but are omitted from the executed binding map. An unbound utility consumer can execute without fake parameter-gradient placeholders; it cannot accumulate parameters. Explicit compact consumers retain mode-three provenance and index roots, while ordinary dense bindings still reject compact gradients.

Consumers accumulate only their explicitly named gradients with the supplied scale once. Complete accumulation preflight precedes ping-pong mutation. Producer edges propagate no implicit gradients; endpoint input-only and primary-only policies still belong to the caller's chosen graphs.

All methods require caller serialisation, including `Close` versus consumer execution, accumulation or download. Successful driver operations must complete device work. No raw resident parameter map or producer buffer export is exposed by this API. `DownloadOutputs` returns owned host copies of pinned original source IDs and checks count, extent and finite values.

## Terminal failures

Producer construction/run errors retain a returned lease; partial uncertain native resources are never blindly freed. Consumer errors after producer execution poison the state. Execution, output borrowing, download validation, accumulation and cleanup failures retain the producer and any constructed failed consumer. The runner records terminal state before consumer cleanup, including late pure accumulation admission errors. State publication, update, snapshot, retry and ordinary close are blocked.

Cancellation before beginning a lease performs no work. Cancellation after a producer is live abandons a training boundary and makes it terminal even before the next consumer runs. An unselected producer download also makes the live boundary terminal. A nil caller context rejects the method without device work. `RetainFailure` lets the caller fence errors outside lease methods and preserves the first terminal error.

Later-row failure is not transactional rollback. The test accumulates one row successfully, then fails the next accumulation: the first row's gradient remains, the failed consumer and producer remain live, and the completed step count stays zero. Successful consumers close synchronously before the producer; producer close failure retains the lease. State close succeeds only after a successful lease close.

## Offline numerical and cost checks

Six deterministic 64-element consumers use one pinned producer, current resident weights and scale-once accumulation. The independent F64 reference covers each consumer value and the running gradient sum, including F32 scale constants and every F32 rounding boundary.

| Comparisons | Mean absolute error | RMS error | Maximum absolute error | Absolute limit |
|---:|---:|---:|---:|---:|
| 768 | 3.50735367e-9 | 5.64199957e-9 | 3.4813187e-8 | 2e-6, unchanged |

This test executes seven tapes (one producer, six consumers), six accumulations and six requested consumer downloads. It downloads no producer boundary. Another test downloads one producer boundary explicitly while keeping it live for later use. The row-slice fixture binds a complete six-element producer root and selects each two-element row through vector mode-one slice commands; no backend pointer-offset/sub-buffer API is added.

The tiny parameter-bound producer charges 24 logical bytes and the complete consumer source charges 36. Admission rejects 59 combined bytes before consumer construction and admits 60. The consumer's dead leading constant forces a nonidentity pruning remap. The successful lifecycle closes three consumers followed by the producer, in order `[2,3,4,1]`.

These are fixture accounting and ownership checks. Producer metadata/constants are copied, and conservative full-source ceilings count borrowed inputs. Host graph-copy cost, complete-request transfer reduction, native memory and latency have not been measured. Existing training downloads, replay identities and scopes are unchanged.

## Final offline gates

Window: **2026-10-02 12:32:08--12:33:07 UTC**. GPUs and Go network access disabled, `GOMAXPROCS=2`, tests `-p=1 -parallel=1`. Final commands all exited successfully.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 390 | 1,130 | 0 | 0 |
| Race detector | 39 | 113 | 0 | 0 |
| CPU/mock regression | 848 | 1,416 | 0 | 1 |
| Compiled-artifact follow-up | 39 | 113 | 0 | 0 |

`TestTrainingRandomMPFRComparisonDiagnostic` is the existing opt-in skip; it did not pass. Vet, host build, Linux ARM64/RISC-V builds, model-layout and diff checks passed. Offline CUDA/PTX and GLSL/SPIR-V compilation/validation passed; this slice adds no new kernel.

Both begin wrappers, byte accounting, lease identity, consumer wrappers/binding admission, downloads, retention and close reached 100% scoped coverage. The shared runner reached 98.9%; producer begin admission/construction reached 90.1%. Remaining producer begin capability/extent/unused-binding branches and native execution are unqualified. Race tests cover supported serialised use, not concurrent-close safety.

Evidence: `/workspace/exports/pocket-gpu-training-tapelease-20261002`; script: `/workspace/tmp/pocket-tapelease-gate.sh`. The initial passing run is preserved in `initial-pass/`. Early fixture compile/accounting failures and a malformed generated gate script were corrected before final gates. A read-only delegate judge timed out after 45 seconds and returned no verdict. Independent source approval was not obtained.

## Model integration

The [opt-in integration](pocket-tts-resident-flowhandoff-20261002.md) now creates the producer lease around transformer forward and retains it across diagonal/distillation row consumers. Vector mode-one slices select read-only rows; host audio-hidden/EOS guards remain. Authorised no-edge consumers can execute unrelated backward branches under the same combined live ceiling. The lease closes before final backbone/update; distinct CLI/scope/replay identities and late-row/cancel/close-failure publication fencing are qualified offline. The [integration plan][plan] describes those remaining changes. Native FTZ/resources/performance and new-voice listening require Rui/@default coordination.

[plan]: pocket-tts-resident-flowhandoff-plan-20261002.md
