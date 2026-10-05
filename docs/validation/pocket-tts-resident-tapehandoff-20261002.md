## Read-only resident tape handoff

`ResidentTapeHandoff` keeps a producer tape alive while a consumer borrows its declared output buffers. The consumer reads those buffers without a producer download or host re-upload. Returned consumer outputs can feed explicit accumulation while both owners stay alive. This standalone primitive has PTX and Vulkan preflight adapters; no training scope, CLI option or replay identity uses it yet.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. The staged inference index is unchanged: eight paths and 469 insertions. Native GPU execution, model runs, listening and publication are on hold.

## Admission and storage ownership

`NewResidentTapeHandoff` validates both complete source tapes, including unused constants, then admits one combined logical-byte ceiling before any allocation. It checks declared producer output IDs, consumer constant placeholders, equal element counts, duplicate consumer targets, duplicate producer storage families (including reshape aliases), and conflicts with existing consumer bindings. PTX/Vulkan adapters preflight all required kernels and existing read-only binding IDs, extents, liveness and overlap before producer construction.

The pair copies graph metadata, constant values, edge descriptors and the consumer binding map. Bound native buffers stay caller-owned. Consumer construction occurs only after producer execution and output binding; the complete consumer bindings undergo another liveness/extent/overlap check before allocation. The constructors repeat their normal admission. A late native-buffer alias failure retains the executed producer.

Both tape owners remain live through consumer execution and external accumulation. Successful `Close` closes the consumer first, then the producer. There is no implicit gradient propagation across an edge. Callers must provide their own explicit VJP/accumulation graph and preserve its scale and stop-gradient rules.

All pair methods and borrowed-output uses require caller serialisation. Concurrent `Close` versus `RunDevice`, output access, downloads or accumulation is unsupported. External bindings must stay live and immutable until successful close: the producer reads its bindings during its run, and the consumer reads its bindings during its later run. Kernel modules also outlive the pair. Returned buffers are read-only borrows whose lifetime ends at successful close.

Drivers must complete and synchronise device work before `RunDevice` returns success. Existing PTX dispatch uses `SyncErr`; Vulkan `VkF32Plan.Run` calls `vkSubmitLocked`, whose `vkWaitPendingLocked` fence wait returns success only after completion. An enqueue-only driver cannot safely implement the handoff interface. External accumulation must also finish and synchronise before close.

## Failure retention

Producer execution, output borrowing, late consumer admission/construction, consumer execution, native output/download and close errors make the pair terminal. It rejects retry and output access without calling the underlying driver, and refuses ordinary close. An error returned with a non-nil partially constructed pair requires the caller to retain that owner for coordinated recovery; blind unwinding of uncertain native work is forbidden.

For work outside the pair, call `RetainFailure(err)` before cleanup on any accumulation error involving a borrowed consumer output. It preserves the first terminal error and both tape owners. The mock test injects an external accumulation failure, then verifies blocked close, retry and output access. No automatic recovery or rollback is implemented. A context cancelled before execution leaves the unexecuted producer safely closeable.

## Numerical and cost checks

Six deterministic 64-element cases compare both consumer values and explicit `previous + .5*contribution` accumulation against independent F64 arithmetic. Inputs and scale constants use their exact F32 values; the reference includes both tapes' reduction rounding.

| F64 comparisons | Mean absolute error | RMS error | Maximum absolute error | Absolute limit |
|---:|---:|---:|---:|---:|
| 768 | 6.0814552e-9 | 8.28276301e-9 | 5.49223511e-8 | 2e-6, unchanged |

The test performs no producer or consumer download during the handoff or accumulation. A separate selected-output test performs exactly one requested consumer download and no producer download. Mutation tests cover copied constants, commands, outputs, edges and the consumer binding map. Lifecycle tests cover premature/closed output access, reverse-order close, terminal access, construction/run/output/download/close failures, and successful accumulation while both owners remain live.

The two tiny graphs each charge 24 logical bytes. Admission rejects a combined limit of 47 bytes without constructing either owner, and admits 48 bytes. Borrowed placeholders are deliberately charged twice in this conservative bound. Metadata and constants are copied to freeze the deferred graph; host graph-copy cost has not been benchmarked. Neither the combined bound nor these mocks measure native peak memory, latency or a complete-request transfer reduction. Existing training scopes retain their downloads and budgets.

## Offline gates

Final gate window: **2026-10-02 12:00:49--12:02:33 UTC**. GPUs were disabled, network access disabled for Go, `GOMAXPROCS=2`, with `-p=1 -parallel=1`. All final gate commands exited successfully.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 140 | 370 | 0 | 0 |
| Race detector | 14 | 37 | 0 | 0 |
| CPU/mock regression | 831 | 1,365 | 0 | 1 |
| Compiled-artifact follow-up | 14 | 37 | 0 | 0 |

`TestTrainingRandomMPFRComparisonDiagnostic` is the existing opt-in skip. It did not pass. Vet, host build, Linux ARM64/RISC-V builds, model-layout check and `git diff --check` passed. Offline `nvcc -ptx -arch=sm_80 --fmad=false`, `ptxas`, GLSL compilation and `spirv-val` passed; the handoff introduces no new shader or kernel.

Scoped coverage: constructor 96.7%; graph copy, run, output, downloads, external retention and close 100%; PTX preflight 100%; Vulkan preflight 93.9%. The uncovered Vulkan binding/lock failures and actual GPU execution are unqualified. Race gates exercise supported caller-serialised use; they do not establish a concurrent-close contract.

Evidence directory: `/workspace/exports/pocket-gpu-training-tapehandoff-20261002`. Script: `/workspace/tmp/pocket-tapehandoff-gate.sh`. The interrupted preceding run is preserved under `interrupted-1150/`; its missing end marker is not a successful final gate.

An earlier abstract lifetime judge requested device completion, explicit external ownership and a no-concurrent-close contract; those requirements are now stated in source. Source review attempts failed on delegate attachment confinement and a 120-second timeout. No independent source approval was obtained.

## Next integration

An opt-in training scope still needs coordinator-owned borrowing from the transformer producer through flow consumers and gradient accumulation, complete source/inventory admission, existing host overflow guards, and terminal publication fencing. This primitive alone removes no training download. Dense persistent gradient/optimizer work, EOS diagnostics, full snapshots, native FTZ/resource/performance checks and new-voice listening are unfinished.
