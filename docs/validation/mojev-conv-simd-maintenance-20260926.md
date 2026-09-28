# MoJev convolution SIMD qualification

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

## Maintenance interruption and completed rerun

The released race command completed `TestMoJevAcceleratedReleased` in147.85s
(short/isolation/ownership/concurrency, hidden maximum9.91821e-5), then entered
`TestReleasedGroupedTextScorer`. Maintenance coordinator gracefully terminated
its process group1543480 before that grouped race completed. The whole command
has no verified success status. Its partial `released-race.log` is preserved
as interrupted evidence. No further tests were started after the maintenance
instruction until Rui explicitly resumed work.

On September26 after the authorised restart, the same grouped race completed
on `298161d2` (which retains convolution commit `afb61da5` unchanged). The existing
approved revision `0c8695b6252f4205907433d4e196a94f032e60c3` was restored to
`/dev/shm/mojev-checkpoint` after the reboot cleared RAM storage. All four
config/weight/tokenizer hashes match the pre-maintenance assets. No new model
revision or numerical fixture was introduced.

`TestReleasedGroupedTextScorer` passed under `-race` at capacity512 in365.97s,
exit0. It checks the4096-total-token reference, sampled grouped hidden rows,
reordered/changed requests running concurrently and retained-output ownership.
Maximum errors remain `1.4007092e-6` logits, `4.6472996e-7` changed-candidate
logit and `5.7220459e-5` hidden, with unchanged `3e-4` / `2e-3` gates. Live heap
after concurrency was72,184 bytes below warm baseline. Process peak RSS was
11,124,404 KiB including race instrumentation; this is not ordinary inference
admission evidence or a new memory-improvement claim.

Post-boot environment: kernel6.8.0-142, Go1.26.3/Linux amd64, i7-12700,
GOMAXPROCS6. NVIDIA was explicitly disabled; no GPU kernel, frozen evaluation
or model service was started. The post-boot driver upgrade does not qualify
GPU stability. Initial host load/pressure and boot identity are preserved.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_MOJEV_GROUPED_BACKEND=simd \
  GO_PHERENCE_MOJEV_GROUPED_CAPACITY=512 \
  GO_PHERENCE_MOJEV_GROUPED_REPORT=OUTPUT.json \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestReleasedGroupedTextScorer$' \
  -v -count=1 -timeout=900s
```

Post-boot evidence: `/workspace/tmp/mojev-conv-postboot-20260926`. This closes
the interrupted CPU grouped-race gate. Native foreign execution, held-out
quality/calibration, hours-long retention and GPU reliability remain open.
`RuntimeReady=false`.
