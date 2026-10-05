# Pocket TTS resident parameter-copy removal

## Status

Resident tape construction now borrows named host parameter arrays read-only instead of cloning them before replacing the tensors with resident device bindings. Public tape builders retain owned parameter copies. Commands, tensor shapes, output/binding IDs, arithmetic, replay scopes and logical admission ceilings are unchanged.

HEAD remains `a90aec30567d485784f2e04d1803c543c9f2817f`; training changes are uncommitted. Eight staged inference paths / 469 insertions remain unchanged. Both GPU backends were disabled throughout validation. No model, native GPU, recovery, listening or publication run occurred.

## Contract

Private flow, transformer, weighting/sample-weighting, projection, audio-assembly and sequence builders accept an internal `borrowParameters` construction flag. Resident synchronous evaluation selects it; public builder wrappers do not.

Only named parameter constants borrow source storage. Inputs, seeds, nonparameter constants and zeros remain owned. Full source values, parameter binding inventory and conservative logical byte ceilings validate before pruning and dispatch. Borrowing does not omit, zero or bypass admission of host parameters, even though native parameter buffers substitute for them later.

Tape construction and execution remain single-owner/non-concurrent. Parameter source arrays stay reachable through the tape; they must not be mutated during admission/execution. Device bindings and uncertain-owner retention are unchanged. No borrowed parameter array is returned as an owned result. Public decoded results and public source-tape ownership remain unchanged.

The flow builder avoids its eager parameter clone; the transformer-style builders mark parameter tensor IDs and avoid only those materialisation copies. No arithmetic folding, reassociation, tensor omission or graph reuse was added.

## Validation

Evidence: `/workspace/exports/pocket-gpu-training-paramborrow-20261002`.
Gate window: **2026-10-02 02:49:03–02:49:44 UTC**.

| Gate | Passed |
| --- | ---: |
| New cases repeated (`-count=10`) | 20 top-level + 80 subtests |
| Same cases with race detector | 2 top-level + 8 subtests |
| Affected host/mock regressions | 135 top-level + 189 subtests |

Final logs contain zero failures/skips. Eight builder cases cover host-time flow, device-time flow, transformer, weighting, sample weighting, projection, audio assembly and sequence. Tests compare command/output/binding metadata, all tensor shapes/constants and interpreted outputs between owned and borrowed paths. They prove private parameter provenance, public parameter independence and continued ownership of nonparameter constants. Full source byte ceilings and nonfinite named parameters are rejected in borrowed mode. Existing pinned numerical tolerances were unchanged.

Regression coverage includes component VJPs, full-step/all-gradient/update parity, objective endpoints, padding, clipped multistep/resume, replay identities, complete binding admission and failure retention. Targeted `constantValues` coverage is 100%; other builder function coverage is recorded in `coverage-functions.txt`. This is host/mock coverage, not native execution.

Affected-package vet, `go build ./...`, Linux ARM64/RISC-V pockettrain builds, model-layout and diff checks passed. No kernel changed, so previous compiler gates were not repeated.

### Allocation evidence

Isolated weighting builder with a 256-by-256 middle matrix, three 100-iteration measurements:

| Construction | Bytes/op | Allocations/op | Observed time |
| --- | ---: | ---: | ---: |
| Public owned parameters | 301,985–301,996 | 142 | 185–208 µs |
| Private borrowed parameters | 34,876–34,877 | 138 | 159–160 µs |

These results measure host construction only. They do not establish native request throughput, peak RSS, device allocation reduction or fewer transfers. Graph metadata and nonparameter constants still materialise before pruning; tensor/command counts and logical byte ceilings remain unchanged.

### Corrections and review limits

Development scripts initially mismatched existing helper signatures and constant variable names; partial edits were completed before final gates. The initial test fixture used an incorrect input-linear field and omitted the sequence prefix BOS row; both were corrected. Initial logs are retained.

The self-contained design judge timed out. No independent source approval or native arithmetic/resource/performance/voice qualification is claimed.

## Remaining work

Host graph metadata and nonparameter constants, dense embedding VJP, live device allocations, broader boundary fusion, inter-tape transfers, diagnostic synchronization and full snapshots remain. Native qualification and new-voice/listening acceptance require coordinated release of existing holds.
