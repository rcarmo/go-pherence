## Retained prediction handoff

`ptx-resident-predictionhandoff-hybrid-v1` and `vulkan-resident-predictionhandoff-hybrid-v1` retain private diagonal/distillation forward tapes as children of the transformer/EOS lease. Backward consumers read explicit forward roots and the transformer condition through read-only device bindings. Original host vector downloads and overflow guards remain.

CLI names omit `-v1`; hash kind is `resident-predictionhandoff-backend`. The new scope inherits EOS admission, forward handoff, diagonal/distillation admission and compact embedding. Public component calls and older scopes/replay identities remain unchanged. Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`; staged inference remains eight paths / 469 insertions. Native GPU/model/audio/listening/publication holds are unchanged.

## Forward child and backward roots

The private admitted forward path binds the same transformer condition root, then constructs a read-only child with all original outputs and its scalar guard pinned. It downloads the same host arrays for existing ownership/finite/overflow validation. The backend records the exact child, row, kind, original root IDs and host output slice identities. Forward child construction never accumulates parameters.

Diagonal backward records private graph IDs for input, desired and computed prediction. Endpoint/primary builders similarly record original input/value IDs; composition remaps the primary input and preserves endpoint velocity, tangent, input, desired and prediction IDs. Those metadata fields add no public arithmetic or graph command. Endpoint tangent/velocity constant creation stays in its original order.

`runPredictionBackward` validates the complete source/binding inventory, kind/row/root tuple, extents, host slice identities, parameter disjointness and constant values. It appends a transformer condition slice and constant placeholders for child roots, then replaces only reads of the explicit IDs. Computed prediction destinations are never rebound; their forward and derivative commands remain in full-source admission. The consumer seeds its loss from the retained identical forward prediction while the original adjoint graph remains intact. Pruning can omit recomputed values only if their derivative paths do not require them.

The multi-source lease runner admits all complete live sources plus the expanded consumer before allocation/pruning. Native liveness/kernel/extent/alias admission checks all parameter, transformer and prediction-child buffers together. Compact parameter roots remain unchanged. Diagonal seeds contain objective scale once and parameter accumulation uses one; distillation exposes primary gradients only and accumulates at the original parameter scale. Endpoint input-only stop-gradient is unchanged; there is no implicit differentiation across the borrowed edges.

After successful backward execution, accumulation and result decoding, the child closes and its roots/host associations are cleared. The root closes only after all row children close, before final reduction/backbone/update. Methods require caller serialisation and successful driver operations must be device-complete.

## Failure fencing

Malformed row/kind/root associations, copied host arrays, mismatched constants, invalid extents, parameter aliases, duplicate IDs, complete-source failures and expanded ceilings reject before backward allocation. Child construction/download or forward guard failure makes the parent tree terminal. Existing host admission precedes all backward work.

Accumulation, cancellation and child close errors retain the prediction child and transformer root. The step returns zero metrics and publishes no update, replay or checkpoint; snapshots, retry and ordinary close fail. Prior successful row gradients are not rolled back. The parent-close helper refuses a live child instead of freeing its borrowed storage. Successful lifecycle tests verify four private children and four backward consumers, with each child closing after its final accumulation.

## Numerical and transfer checks

Pinned independent PyTorch loss, all gradients, updates and EMA pass with unchanged tolerances. PTX/Vulkan tests cover objective endpoints, EOS weights, clipping over multiple steps, padded rows, repeated/empty tokens and voice, per-tape/output failures, replay/resume and cross-scope restore rejection.

An explicit borrowed-storage interpreter fixture observes actual producer arrays in backward constant bindings. It verifies retained forward-root pointer identity and one close after accumulation per child. Legacy mocks keep their prior copied-parameter behaviour unless this lifecycle hook is enabled. Numerical parity and ownership evidence are separate; pointer identity checks binding ownership, not generated numerical equivalence.

| Tiny step | EOS admission | Prediction handoff |
|---|---:|---:|
| Tapes | 11 | 11 |
| Commands | 3,719 | 3,707 |
| Downloads | 47 | 47 |
| Downloaded elements | 122 | 122 |
| Accumulation preflight calls | 161 | 161 |
| Executed logical bytes | 134,256 | 134,240 |
| Planned storage bytes | 42,108 | 42,268 |

Pruning removes 12 recomputed value commands while preserving required derivative paths. Extra read-only roots/slices alter lifetime allocation, adding 160 planned bytes. Retained forward values avoid re-upload into backward; host graphs/constants still exist and every original download remains. The accounting sums tape plans; simultaneous root/child/consumer native memory, allocator overhead and request latency have not been measured. The conservative complete-source live ceiling still charges all retained sources before allocation.

## Final offline gates

Window: **2026-10-02 14:10:03--14:11:39 UTC**. GPUs and Go network access disabled, `GOMAXPROCS=2`, tests `-p=1 -parallel=1`. All final commands exited successfully.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 860 | 1,690 | 0 | 0 |
| Race detector | 86 | 169 | 0 | 0 |
| CPU/mock regression | 941 | 1,585 | 0 | 1 |
| Compiled-artifact follow-up | 86 | 169 | 0 | 0 |

`TestTrainingRandomMPFRComparisonDiagnostic` is the existing opt-in skip; it did not pass. Vet, host build, Linux ARM64/RISC-V builds, model-layout and diff checks passed. Offline CUDA/PTX and EOS-admission GLSL/SPIR-V compilation/validation passed; this slice adds no kernel.

Scoped coverage: scope/child close 100%; child begin 95.7%; backward remapping/admission 93.3%. Malformed partial root/value count and final expanded admission branches are not fully covered. Native arithmetic/resources/performance remain unqualified; race gates cover supported serialised use only.

Evidence: `/workspace/exports/pocket-gpu-training-predictionhandoff-20261002`; script: `/workspace/tmp/pocket-predictionhandoff-gate.sh`. `initial-pass/` preserves the preceding successful broad run. A focused read-only graph/lifetime source judge timed out after 60 seconds without a verdict. Independent source approval was not obtained.

## Remaining downloads

Host hidden, prediction/desired/input and backward results still transfer for guards and dispatch. Removing forward vector downloads requires a separate scalar admission/result contract and builders that can consume declared device inputs without fabricated host numerical values. Dense persistent optimizer work, snapshots and graph construction remain. Native FTZ/resources/throughput and new-voice listening require Rui/@default coordination.
