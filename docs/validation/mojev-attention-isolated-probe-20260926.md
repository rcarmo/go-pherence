# MoJev isolated attention diagnostic

`TestMoJevPTXAttentionIsolated` prepares a single named attention launch, a host
oracle and output guards. It is diagnostic infrastructure; the GPU path has
**not been executed** since the [bus-loss incident](mojev-gpu-bus-loss-audit-20260926.md).
No GPU recovery, reset or change to main inference dispatch accompanies it.

## Isolation and admission

The probe requires its own `GO_PHERENCE_MOJEV_ATTENTION_PROBE=1` switch. The
ordinary `GO_PHERENCE_MOJEV_NVIDIA=1` switch does not enable it. Without the
explicit diagnostic switch, even direct selection of the test skips before
any CUDA access. Configuration validation and host reference computation occur
before module loading or device allocation.

| Suffix after `GO_PHERENCE_MOJEV_ATTENTION_PROBE_` | Default | Accepted values |
|---|---|---|
| `KERNEL` | `compact` | `compact` (`mj_attention`) or `tree` (`mj_tree_attention`) on main |
| `ROWS` | `3` | 3–512; values above33 additionally require `LARGE=1` |
| `STATE` | `1` | positive, less than rows |
| `QUESTION` | `1` | positive; state+question must leave a nonempty candidate |
| `LARGE` | unset | unset/0, or1 to explicitly admit more than33 rows |

The shape bounds use checked parsing and bounded subtraction to reject integer
overflow before any work. The probe contains no retries, size sweeps, stress
loops or hidden reference-kernel launches. Each invocation launches exactly
one attention entrypoint. Run only that test once in a fresh process when an
authorised recovery investigation permits it; a test runner can otherwise repeat
a selected test with `-count`, which is outside this one-invocation contract.
The extra large-shape flag is an admission check, not proof those shapes are safe.

## Oracle and buffers

Eight query heads share two K/V heads, each256 elements wide. The deterministic
small finite Q/K/V inputs are compared with an independently expressed float64
host attention calculation: stable softmax over visible rows and sigmoid gate.
State rows see the state, question rows see state+question, and candidate rows
see the whole single branch. The tree mode has one candidate; this probe does
not test sibling isolation. Existing ordinary tree tests cover that separately.

The numerical gate remains the synthetic attention absolute tolerance `1e-6`.
Host self-tests cover uniform weights and a nonuniform closed-form case with
scores0/1/2, GQA head mapping and all three visibility boundaries. This is a
synthetic diagnostic, not released-model or task-quality evidence.

Output storage has17 leading and19 trailing F32 sentinels. The kernel receives
an offset device pointer; the entire allocation is downloaded to check active
output and bitwise unchanged guards. Each launch logs its actual kernel, rows,
state/question dimensions, grid and block geometry. An explicit `SyncErr`
separates launch completion from download errors.

Cleanup closes/synchronises the module before freeing buffers. If module cleanup
fails, the test reports the failure and leaves buffers owned until process exit
rather than freeing storage which outstanding work might still use. This is
intentional error containment, not a successful cleanup or leak-free run. No
automatic reset or inference retry follows an error.

## Verification and use

CPU-only configuration/oracle/input tests pass under race ten times. The
hardware test explicitly skips. Vet and full build pass; MoJev test binaries
cross-build for Linux ARM64/RISC-V. Foreign binaries were not executed. Go
statement coverage does not instrument these helpers because they are test-file
code; the filtered run's package coverage is not a measure of diagnostic coverage.

A focused read-only review found no concrete safety bug. The caller-config
integer-overflow edge found during implementation was fixed and regression-
tested. Review also noted retained buffers on module-close failure; the contract
above describes that intentional safety trade-off. Test support is ready for a
future isolated investigation, but hardware behaviour remains unverified.

Model-free verification only:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./model/mojev \
  -run 'AttentionProbe|PTXAttentionIsolated' -count=10 -v
```

Before enabling hardware execution, satisfy the recovery/telemetry prerequisites
in the bus-loss audit. Keep guest/physical-host PCIe/Xid logs and GPU telemetry
running throughout the process. Begin with the default three-row compact
reference in one process, then the three-row tree case in another. Stop at the
first CUDA error, failed device query or new Xid; do not jump to512 rows or rerun
the broad sanitizer suite. The retained candidate branch can extend this same
probe with its warp kernel, without running the compact reference in that process.

Evidence: `/workspace/tmp/mojev-attention-isolated-20260926`. This change adds no
new inference optimisation or proven GPU-crash fix. `RuntimeReady=false`.
