## Shape-only external tape inputs

`TensorSpec.External` declares a read-only device input by shape, with `Values=nil`. It lets graph admission and native borrowing represent an actual device source without fabricated host numerical arrays. No training scope uses shape-only inputs yet, and all existing host guards/downloads remain.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths / 469 insertions. GPU/model/audio/listening/publication holds are unchanged.

## Admission and ownership

`Tape.Validate` treats an external tensor as initialized and read-only. It requires valid extent, charges every external extent to the full logical ceiling, forbids host values on the same descriptor, and rejects commands writing that input. Shape validation alone supplies no finite-value, index-provenance or overflow approval; runtime guards remain necessary.

`ValidateExternalTapeBindings` requires every declared external source to be explicitly bound before execution, including dead sources before pruning. It rejects bound writable outputs and invalid IDs. PTX/Vulkan bound constructors invoke it before allocation and repeat backend exact-extent/liveness/output-alias admission. Handoff adapters admit external sources along with ordinary host constants. Borrowed external storage is never allocated/uploaded or freed by the tape.

The shared resident runner validates all full-source external bindings before pruning. Child/multi-source lease targets accept shape-only sources; all existing identity, complete live-source ceiling, parameter-inventory and terminal-retention checks remain. Ordinary unbound tapes and root leases with no device edge route reject missing external sources before construction. Named resident parameters retain their old explicit host-constant inventory; this slice does not add an externally supplied parameter map.

Pruning copies the external declaration with stable IDs and owned shape/operand metadata. Lifetime allocation already pins all unwritten source families; external reshape aliases share that pinned family and never enter positive-zero pooling. Constants requiring host provenance -- embedding indices, EOS labels, time inputs and LSD cutoffs -- still require explicit host values. An external numeric input cannot replace them to bypass validation.

`WithExternalTapeInputs` first validates the complete original source, deep-copies graph metadata/constants, then removes only explicitly named host constants and marks them external. It revalidates the result. Duplicate/invalid IDs, computed destinations and unused nonfinite constants reject; caller graphs remain owned and unchanged. This helper provides a source contract, not numerical equivalence or permission to remove a guard.

All existing builders produce `External=false`. No CLI/hash/replay identity or model arithmetic changes. The offline interpreter requires all external bindings and uses the actual read-only arrays when supplied, rather than copying or manufacturing source values.

## Offline ownership and numerical checks

Tests cover missing live/dead bindings before pruning, invalid source extents, attempts to write externals, host-values conflicts, explicit root outputs, reshape aliases, zero pooling, complete-source conversion/mutation ownership and required host metadata. A retained child consumes an external root directly; a later consumer borrows its output, with no host source array or producer download.

Borrowed-only PTX/Vulkan fixtures construct one external-root tape with no owned allocations/plans and no kernel dispatch. Close leaves the source buffer/tensor live. These use offline fake metadata; no native device was opened. Backend handoff tests exercise complete extent/liveness/alias and missing-binding admission.

512 independent F64 comparisons use exact F32 external inputs/constants/scales over eight consumer scales, retaining the producer. Only requested consumer outputs are downloaded and no gradients accumulate.

| Comparisons | Mean absolute error | RMS error | Maximum absolute error | Absolute limit |
|---:|---:|---:|---:|---:|
| 512 | 4.24041355e-9 | 6.33230426e-9 | 2.79396772e-8 | 2e-6, unchanged |

The 64-element producer plus three-tensor consumer charges 1024 logical bytes. External sources remain in conservative admission, even when the underlying owner is borrowed. Native memory, host graph allocation savings and complete-request latency have not been measured.

## Final offline gates

Window: **2026-10-02 14:27:37--14:29:02 UTC**. GPUs and Go network access disabled, `GOMAXPROCS=2`, tests `-p=1 -parallel=1`. All final commands exited successfully.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 680 | 1,280 | 0 | 0 |
| Race detector | 68 | 128 | 0 | 0 |
| CPU/mock regression | 953 | 1,585 | 0 | 1 |
| Compiled-artifact follow-up | 68 | 128 | 0 | 0 |

`TestTrainingRandomMPFRComparisonDiagnostic` is the existing opt-in skip; it did not pass. Vet, host build, Linux ARM64/RISC-V builds, model-layout and diff checks passed. Offline CUDA/PTX and GLSL/SPIR-V compilation/validation passed; no shader or kernel changed.

Scoped coverage: `Tape.Validate` and both external contract helpers 100%; pruning 98.1%; PTX/Vulkan handoff adapters 96.3%/91.4%. The new missing-binding checks and borrowed-only construction paths are exercised. Native execution/allocator/resource qualification and some adapter error branches are unqualified. Race gates exercise supported serialised ownership only.

Evidence: `/workspace/exports/pocket-gpu-training-externalinputs-20261002`; script: `/workspace/tmp/pocket-externalinputs-gate.sh`. `initial-pass/` preserves the preceding successful broad run. An initial validation fixture used a nonexistent variable name and was corrected before the passing gates. A focused read-only source judge timed out after 60 seconds with no verdict; independent source approval was not obtained.

## Next model contract

Private shape-only backward builders still need explicit extent/owner/admission descriptors and scalar finite/overflow roots, without substituting zero arrays into existing host numerical guards. Preserve complete source/binding/combined-live admission, primary-only gradients and endpoint stop-gradient. Existing hidden/prediction/desired/input/backward downloads and full snapshots remain; independent source/native qualification and voice listening are still required.
