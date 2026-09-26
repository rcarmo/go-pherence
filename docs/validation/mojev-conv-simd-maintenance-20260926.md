# MoJev convolution SIMD maintenance checkpoint

CPU four-tap convolution now uses separate existing SIMD multiply/add operations,
reusing idle projection padding scratch. Scalar-only dispatch retains the old
loop. No new allocations or FMA rounding changes are introduced. GPU execution
remained disabled throughout this work because of the unresolved bus-loss incident.

Before maintenance, verification completed:

- Bitwise comparison of all6144 channels with scalar reference, native and
  disabled SIMD, negative zero/subnormals, guards, source preservation, sibling
  ancestry isolation and zero warm allocations.
- Ordinary released short/isolation, independent512-path and4096-total grouped
  reference tests passed with unchanged gates.
- Whole-tree NVIDIA-disabled race exited0; affected races, vet/build and
  ARM64/RISC-V builds/test-binary cross-builds passed.
- New helper statement coverage100%. Focused read-only review found no bounds,
  rounding, scratch-lifetime or worker-race issue in its scope.
- Ten normal-value microbenchmark samples:19.743 →3.804 microseconds per row,
  −80.73% (p<.001), zero allocations. Subnormal stress is kept in correctness
  tests and measured separately, not used as the normal-value speed claim.
- Six interleaved process pairs, with one-second host CPU-pressure/throttling
  snapshots: all48 workload examples exact. Full-request latency geomean1.59%
  lower, individual differences statistically inconclusive. Keep this measured
  local improvement for cumulative benefit; no established whole-model speedup.

Baseline `cda5a1b2`, Go1.26.3/Linux amd64, i7-12700, GOMAXPROCS6, NVIDIA disabled.
Approved existing checkpoint hashes checked before load, one checkpoint process
at a time. Data: `/workspace/tmp/mojev-conv-simd-20260926`.

## Interrupted gate

The released race command completed `TestMoJevAcceleratedReleased` in147.85s
(short/isolation/ownership/concurrency, hidden maximum9.91821e-5), then entered
`TestReleasedGroupedTextScorer`. Maintenance coordinator gracefully terminated
its process group1543480 before that grouped race completed. The whole command
has no verified success status; **the new grouped race is unverified**. Its
partial `released-race.log` is preserved. No further tests were started after
the maintenance instruction.

This is a maintenance checkpoint, not full qualification. No GPU reset/recovery
or new GPU testing was attempted. Pause all work and do not auto-resume until
Rui asks. Native foreign execution, held-out quality/calibration, hours-long
retention and GPU reliability remain open. `RuntimeReady=false`.
