# MoJev CPU contexts through 4096 tokens

## Scope

The native SIMD scorer now accepts an explicitly configured capacity from3 to
4096 tokens. Previously both accelerated backends stopped at512. NVIDIA still
stops at512; no GPU kernel, service or frozen evaluation was run for this work.
Baseline is `de9934b9`, retaining the adopted convolution, padding, GEMM and memory
optimizations.

The limit is **tokens in one packed tree invocation**, not an independent budget
for every sibling. Candidate groups stay within the configured capacity. The
public scorer's existing4096-total-token request budget,64-candidate limit and
minimum two candidates remain unchanged. Consequently its longest possible
candidate path is4095; the encoder-only test separately reaches position4095 in
a4096-token path. The checkpoint's16384-token metadata context is not qualified.

This removes an implementation limit, not a model-quality defect. JevBench
results remain pinned to the prior512-capacity implementation. No benchmark
prompts, labels, thresholds or recorded results were changed or rerun here.
Structured JSON states, images and broader quality still require separate work;
`RuntimeReady=false`.

## Implementation

- `ValidateQwen35F32Branch` retains its public3–512 contract, used by NVIDIA.
  The new CPU preflight `ValidateQwen35SIMDBranch` invokes the same geometry and
  finite-weight checks with `Qwen35SIMDMaxTokens=4096`. CPU construction uses that
  preflight, so callers need not guess which validator admits longer contexts.
- Four branch metadata arrays (starts, stops, positions, parents) now have
  constructor-sized storage. They are reused only while the existing branch
  context mutex is held, and every active entry is rewritten per invocation.
- State/question readout masks and64 candidate-mask rows are similarly sized to
  the scorer capacity and protected by its existing mutex. Active slices are
  cleared before use, including after different tree shapes.
- Constructor bounds and per-scorer capacity remain checked before execution.
  Scores/destination publication remains transactional. No per-forward metadata
  allocation, arithmetic change, widened tolerance or new GPU limit is added.
- Full attention still uses a single score vector, not a materialized L² matrix.
  Its computation grows with attention visibility, so maximum-size requests are
  much slower despite bounded, approximately linear scratch.

An initial ordinary released test caught an old `[512]bool` readout-mask access
at index512. That failure is preserved as `released-mask-bound-failed.log`; the
mask storage was corrected before the successful ordinary and race runs. It is
not counted as a pass. A later external interruption stopped the64-candidate
race before results; its log is retained separately from the completed rerun.

## Independent fixtures

`scripts/mojev_oracle_cpu_context.py` uses pinned upstream MoJev and the approved
weights without consuming Go outputs or benchmark labels. Two offline runs
produced byte-identical `testdata/native_cpu_context.json`.

- Source revision: `a74d58cd19ec573e83e8e27f9fecd837b8d830fb`.
- Checkpoint revision: `0c8695b6252f4205907433d4e196a94f032e60c3`.
- Pinned `mojev/modeling.py` SHA:
  `a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458`.
- Safetensors SHA:
  `eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
- CPU-only Torch2.14.0+cpu; Transformers5.17.0, Qwen module SHA
  `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
- Generator SHA:
  `e4f4ba0ff7240806dc023bad215f70167b26134d2a9599ffcc0e588c1b4607f5`.
- Fixture SHA:
  `495d452b956ca618fbdd35424fcbba0a29cfc047db9c923ccb1c5412163382e8`.

The fixture contains513- and1024-token paths, a4096-total-token public request
with two4095-token paths, an unequal-sibling4096-token tree (1792/3072-token
paths), and a4096-token encoder-only path. Each candidate is evaluated separately
with fresh state and local positions by the independent reference. Hidden rows
are sampled at the last state, question and candidate positions. The weights
are the original released BF16 values upcast to F32 under the repaired text
policy, not a new checkpoint or the leaky packed upstream policy.

The Python environment was recreated outside the repository with CPU-only
Torch; `CUDA_VISIBLE_DEVICES=''` and offline asset loading were set. No new model
weights were downloaded. Existing fixture files were not modified.

## Correctness gates

On Go1.26.3/Linux amd64, reported i7-12700, `GOMAXPROCS=6`, NVIDIA disabled,
one checkpoint process at a time:

| Gate | Result |
|---|---|
| New long-context ordinary released test | PASS678.80s |
| Same long-context test under race | PASS1586.00s |
|64-candidate4096-token tree under race | PASS473.80s |
| Whole-tree CPU race,189 packages | Exit0, wall5m08.92s |
| Affected-package races | Passed |
| Vet/build, docs/layout | Passed |
| Linux ARM64/RISC-V runtime/Qwen/MoJev cross-builds | Passed; not native execution |

Maximum new-reference errors are `7.748603820800781e-6` logits and
`2.4509429931640625e-4` hidden, within unchanged `3e-4` / `2e-3` gates.
The64-candidate grouped fixture remains at `1.4007091522216797e-6` logits,
`4.647299647331238e-7` changed logits and `5.7220458984375e-5` hidden.

Tests cover exact unrelated-sibling stability, candidate permutation, grouped
hidden rows, maximum encoder position,4097-token rejection without output,
active cancellation followed by deterministic recovery, two concurrent callers,
and retained output ownership. Model-free tests separately cover constructors,
configured-capacity rejection (512/513,1024/1025,4096/4097), canceled calls before
scratch access, CPU versus unchanged GPU admission, and padding geometry1–4096.
The existing row/candidate validation still enforces the64-candidate axis before
readout masks are accessed; the maximum64-candidate released test exercises it.

Ordinary unit-suite statement coverage is47.8% Qwen and67.3% MoJev. The new CPU
validator wrapper and projection padding helper have100% coverage, but ordinary
unit coverage is low in released-weight constructors/forward paths (for example
`scoreTree`20%). Released tests exercise those paths without coverage
instrumentation. The repository's changed-code coverage target is not claimed
met by these ordinary percentages; no exclusions or tolerance changes conceal
that gap. Native ARM64/RVV execution remains open.

Delegated design/final reviews of supplied code descriptions found no concrete
CPU4096 defect but highlighted the public-validator distinction, stale512
buffers, maximum candidate count and platform/concurrency limitations. The
CPU-specific public validator and64-candidate race address the first and third
points; local inspection resolved the fixed mask bug. Those reviews were not
independent full-checkout audits. Only this fixed24-layer0.8B geometry is admitted.

## Memory and short-request regression check

The ordinary maximum-context run dropped the caller CPU encoder after packing.
Loaded post-GC heap was3,782,289,744 bytes; after mixed reuse/concurrency/cancellation
it was3,782,031,504 bytes. Peak RSS was4,959,916KiB (about4.73GiB), zero swaps,
and two goroutines remained. The corresponding race run peaked at11,496,216KiB.
The separate64-candidate race retains the caller CPU scorer deliberately; its
peak13,168,032KiB is not an ordinary-memory admission figure. Its post-concurrency
heap was77,352 bytes below warm baseline.

This is a memory-capacity tradeoff, not a memory reduction. For amd64 at4096,
constructor numeric scratch, hidden output, four metadata arrays and readout
masks total about731MiB, excluding Go slice/map headers, row descriptors and
head-local output allocations. The analogous512 footprint is about93MiB.
Choosing capacity256 or512 remains supported and avoids reserving4096 scratch.
No loading-peak improvement or hours-long retention is claimed.

Six interleaved ABBA processes per version, same capacity256 and four existing
short workloads, verify the change does not require larger default allocation.
Each process checks all four checkpoint/tokenizer hashes, warms once and times
five calls per workload. All288 actual responses are exact. Latency geomean is
+0.86%, byte geomean+0.29%, allocation-count geomean+0.18%; none of the individual
comparisons is statistically significant. Peak RSS ranges overlap. This is
short-request regression evidence, not proof of equivalence or a speedup.

The long released tests combine multiple forwards, hidden probes, isolation and
cancellation checks. Their elapsed times are gate durations, not single-request
latency benchmarks. No attempt is made to compare them with previous short
request timings or to claim generative decode/training performance.

## Reproduction and evidence

Evidence: `/workspace/tmp/mojev-cpu-context-20260926`: both independent oracle
outputs and logs, CPU environment, original failed/interrupted logs, successful
ordinary/race/grouped logs with exit status, short-request distributions and
response checks, host data, source overlay, cross-build and coverage records.
The Python environment, clean upstream checkout at `a74d58c` and cross-build
executables were removed from `/workspace/tmp` on 29 September 2026 during
workspace cleanup. The report, source overlay, oracles and raw test evidence
remain in that directory. Recreate tooling from the pinned source and
requirements before rerunning the gates.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CPU_CONTEXT=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestReleasedCPUContext$' \
  -count=1 -v -timeout=5400s
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_GROUPED_BACKEND=simd \
  GO_PHERENCE_MOJEV_GROUPED_CAPACITY=4096 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestReleasedGroupedTextScorer$' \
  -count=1 -v -timeout=1200s
```

No GPU qualification, structured-state support, HTTP production service,
checkpoint16384-context support or new JevBench score is implied.
