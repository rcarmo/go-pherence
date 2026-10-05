## Private shape-only flow backward builder

`buildFlowDeviceInputsTrainingTape` builds a reverse-over-forward flow graph with four read-only device sources: condition, input, value seed and tangent seed. It allocates no fabricated host numerical arrays for these inputs. The private builder has no production caller; CLI scopes, download policy and host numerical guards are unchanged.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths / 469 insertions. GPU/model/audio/listening/publication holds are unchanged.

## Source and graph contract

The existing host and private device paths share `validateTrainableFlowHeadSizes` for model topology and input extents. Host builders still check finite condition/input/value and tangent seeds. The device builder admits actual host times, time selection, frequencies, frequency squares and the existing 8192 angle ceiling; it does not approve the numerical contents of external buffers.

`flowDeviceInputSizes` supplies extents. `flowDeviceInputIDs` identifies the four shape-only sources; `flowDeviceTrainingTape.validate` checks their uniqueness, declarations, shapes, input provenance and complete external accounting. These fields carry no device-owner identity or numerical admission. Execution still requires complete retained-lease binding and finite/overflow admission before accumulation.

`buildFlowDualTrainingTapeSources` assembles both paths. Device sources are declared directly with `External=true` and `Values=nil`. Literal-zero input tangents remain actual zero constants. Host time metadata and model parameters remain explicit finite-admitted constants. Private synchronous parameter borrowing is read-only and requires immediate resident rebinding, as in the existing private builders; public builders retain owned parameter copies.

Commands, IDs, outputs, parameter inventory, differentiation and scaling match the existing device-time host-input builder. Forward graph recomputation remains. Full-source logical budget includes all external extents before pruning, and pruning preserves source IDs through stable remapping. No device buffer allocation, peak memory or complete-request timing improvement has been measured.

## Ownership and numerical tests

Tests cover all selected-time choices (-1/0/1), host/device graph equality, external binding, pruning/remapping, all parameter adjoints, caller/graph metadata isolation and private/public parameter ownership. Malformed dimensions/topology, BF16 weights, nonfinite constants/times, ambiguous host/device arguments, duplicate/tampered source descriptors and byte ceilings reject.

A retained-lease fixture binds all four external sources directly from one producer. Only boundary outputs download; no source or parameter-gradient downloads occur. Each parameter accumulates once at scale 0.25. After owner close, update preflight observes the same scaled gradients; boundary derivatives stay unscaled.

The host schedule interpreter initially rebound borrowed roots without propagating them to reshape aliases under storage reuse. Tests exposed incorrect zero inputs. The test-only interpreter now propagates the borrowed slice across its pinned storage family, matching PTX/Vulkan constructors. The resident mock preserves external declarations and supplies borrowed arrays rather than attaching conflicting host values. Production kernels/constructors are unchanged.

Independent pinned PyTorch fixture: `model/pockettts/testdata/flow_head_backward_pytorch.json`, schema 1 and `UpstreamCommit`. Both mixed-AD cases cover output, JVP, condition/input/time adjoints and every parameter gradient.

| Comparisons | Mean absolute error | RMS error | Maximum absolute error |
|---:|---:|---:|---:|
| 786 | 4.23488273e-9 | 1.74892247e-8 | 1.97440386e-7 |

Unchanged absolute gates: output 3e-6, JVP 5e-6, input/condition/time adjoints 8e-6, parameters 1e-5. Numerical comparisons use tolerances; no generated-output hash gate is used.

The tiny graph remains 429 commands / 524 tensors / 11,984 full-source logical bytes. The four external sources charge 36 bytes. Both host and shape-only graphs admit exactly 11,984 bytes and reject 11,983. Native storage and transfers are unmeasured.

## Final offline gates

Window: **2026-10-02 14:54:32--14:56:00 UTC**. GPUs and Go network access disabled, `GOMAXPROCS=2`, tests `-p=1 -parallel=1`.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 610 | 1,340 | 0 | 0 |
| Race detector | 61 | 134 | 0 | 0 |
| CPU/mock regression | 975 | 1,631 | 0 | 1 |
| Compiled-artifact follow-up | 61 | 134 | 0 | 0 |

The existing opt-in `TestTrainingRandomMPFRComparisonDiagnostic` skipped. Vet, host build, Linux ARM64/RISC-V builds, model layout, diff checks and offline PTX/SPIR-V compile/validation passed. Kernels and ABI are unchanged. Race tests use supported serialised ownership.

Scoped coverage: new private helpers and shared topology validator 100%; shared source builder 99.1%. A focused read-only source judge timed out after 120 seconds with no verdict. Independent source/native approval was not obtained. Initial fixture compile errors, the interpreter reshape defect and an incorrect time-shape test bound were corrected before final passing gates; first-pass logs are retained.

Evidence: `/workspace/exports/pocket-gpu-training-flowdeviceinputs-20261002`; gate script: `/workspace/tmp/pocket-flowdeviceinputs-gate.sh`.

## Next integration

Retained-owner/row/root/seed provenance and scalar finite/overflow admission still need a private execution contract before composing diagonal or distillation backward graphs. Keep primary-only accumulation, scale-once and endpoint stop-gradient, all-live combined ceilings and terminal publication fences. Hidden/prediction/desired/input/backward downloads, coherent snapshots and dense persistent optimiser work remain. Native qualification and voice listening have not run.
