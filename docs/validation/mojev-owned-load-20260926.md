# MoJev owned-buffer weight loading

MoJev now transfers decoded F32 weights into tensors instead of copying the
entire payload again. CPU loading allocates about 3.01 GB less; the measured
SIMD process peak RSS falls from 5.78–5.90 GiB to about 4.76 GiB. Steady-state
weights and responses are unchanged.

## Ownership

`weights.Source.GetFloat32` returns owned converted arrays. `mojevTextSource.Get`
already validates dtype, raw/decoded shape, element count and finite values.
After those checks it now calls `tensor.FromOwnedFloat32`, which adopts the data
without copying. The caller relinquishes access to that slice. The original
`tensor.FromFloat32` remains a copying constructor and all its callers retain
that behaviour. Tensor shape metadata is still copied.

`GetRaw` remains borrowed and is used only for metadata checks. The owned data
is independent of the source mapping. Source closure and GC do not invalidate
the tensor. The source contract is required: an implementation returning reused
or mapped data from `GetFloat32` would violate the existing owned-array contract.
MoJev does not rely on ownership of the source's returned shape slice; it checks
it and constructs from separately copied requested dimensions.

Tests verify same-backing adoption, scalar/empty/multidimensional geometry,
shape-copying, mismatch/overflow rejection, GC lifetime, lazy arithmetic and
unchanged copying semantics of `FromFloat32`. A source double records a weak
pointer to its owned conversion, destroys its original data/shape on close,
and verifies that the tensor still has the conversion data after GC.

## Measurement

Baseline `6246a7cc`; Go 1.26.3/Linux amd64, i7-12700, `GOMAXPROCS=6`, NVIDIA
disabled for the load/memory probe. One checkpoint/backend process runs at a
time. The probe verifies config, weights and tokenizer hashes before loading.
Approved weight SHA-256:
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.

Three runs per version, with before/after binaries preserved. Hash verification
precedes measurement and warms file pages; these are not cold-disk timings.
CPU load includes config/readout and decoded text weights. Preparation also
loads the tokenizer and creates the SIMD scorer at capacity256. Requests use
the existing four workloads, one warm-up and five calls each.

| Measurement | Baseline range | Owned-transfer range |
|---|---:|---:|
| CPU load cumulative allocation, bytes | 6,792,205,496–6,792,206,328 | 3,782,624,584–3,782,625,256 |
| Through SIMD preparation allocation, bytes | 8,951,542,536–8,951,596,032 | 5,942,013,360–5,942,178,448 |
| CPU load elapsed | 3.089–3.124 s | 2.355–2.375 s |
| Through preparation elapsed | 4.395–4.463 s | 3.700–3.762 s |
| Process peak RSS, KiB | 6,056,300–6,191,564 | 4,985,728–4,991,872 |

After GC, live heap stays about 5.124 GB with both scorers, 3.134 GB after dropping
the original CPU scorer, and 3.066 GB after requests. This change reduces the
copies and transient loading overlap; it does not remove more persistent weights.
All four responses match exactly in each comparison. No inference-latency
improvement is attributed to this loader-only change.

The four interleaved repeat processes record host snapshots every second:
`/proc/stat`, load average, CPU pressure, cgroup quota/throttling, affinity and
available CPU frequency, plus a process list at start. All recorded cgroup
throttling deltas were zero. CPU pressure was nonzero (initial `some avg10`
3.98–13.78), so elapsed loading time is descriptive, not a contention-controlled
speed claim. The allocation reduction reproduced independently of that noise.
The first pair predates this collector; raw files distinguish those runs.

## Qualification

- Tensor and MoJev model-free races pass ten times, including the source-close
  test. New constructor statement coverage is100%; the surrounding MoJev tensor
  loader is88.5% in default tests. No coverage exclusion was added.
- NVIDIA and SIMD lifetime tests pass with the pinned checkpoint under race,
  sequentially. Both verify caller CPU scoring, original-encoder collection and
  accelerated scoring after GC. CPU/GPU/SIMD base-logit errors are respectively
  `1.28150e-6`, `2.20537e-6`, `3.69549e-6` against the unchanged `3e-4` gate.
- Whole-tree NVIDIA-disabled race exits0. Vet/build and Linux ARM64/RISC-V builds
  plus tensor test-binary cross-builds pass. Foreign binaries were not executed.
- Focused read-only review found no runtime ownership/GC issue. It noted that
  shape ownership in the general source API is not explicit; this path does not
  retain or depend on those shape slices. No unrelated API contract was changed.

The initial delegated test draft contained an invalid slice assignment; it was
removed before tests ran successfully. The implementation does not change tensor
arithmetic, reference fixtures or numerical tolerances.

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./tensor ./model/mojev -count=10
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s

GOMAXPROCS=6 GO_PHERENCE_MOJEV_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestNVIDIATextScorerHostLifetime$' -v -count=1
# Separately: disable NVIDIA, set GO_PHERENCE_MOJEV_SIMD=1,
# and run TestSIMDTextScorerHostLifetime.
```

Evidence: `/workspace/tmp/mojev-owned-load-20260926` contains probe source,
paired host collector, JSON responses, phase allocation/RSS logs and test logs.
Native foreign inference, held-out quality/calibration and hours-long service
retention remain open. `RuntimeReady=false`.
