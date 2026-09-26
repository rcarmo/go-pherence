# MoJev maximum-total grouped reference

Both accelerated scorers now match an independent 4096-total-token reference
at scratch capacities256 and512. Two concurrent maximum-total requests also
pass under the race detector on each backend. This is a test/evidence follow-up
to `8cdd4d2f`; runtime code and numerical tolerances are unchanged.

## Input and independent oracle

The request has64 state tokens, one64-token question and64 candidates of62
tokens: `64 + 64 + 64×62 = 4096`. Every individual path is190 tokens. This is
**not** a4096-token path and does not increase the512-token accelerated limit.

`scripts/mojev_oracle_grouped_text.py` executes each candidate separately through
the pinned upstream Python scorer with fresh history and branch-local positions.
It uses existing approved BF16 weights upcast to F32; no Go scores participate.
It stores all64 logits and hidden rows for candidates0/31/63 at local
positions63/127/189. It independently scores a changed final candidate and
reruns the first candidate afterward to check its history remains fresh.

Two offline runs produced byte-identical JSON. Pins:

- Upstream `a74d58cd19ec573e83e8e27f9fecd837b8d830fb`.
- Model revision `0c8695b6252f4205907433d4e196a94f032e60c3`.
- Weights SHA-256 `eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
- Generator SHA-256 `4eeaa8a8c74ca5b3a8134bace2c5939c5ff6e7b8bdccc3999dc03c3ddd1e8428`.
- Fixture SHA-256 `a35a5b2c808793ffc7db9611d981233a9ec5a9db003f2815839d66492d4ca9bd`.
- Transformers5.17.0 implementation hash and Torch2.14.0+cu130 are checked by the
  generator and fixture test. The reference uses CPU execution with two threads.

The repaired fresh-branch policy remains distinct from upstream's leaky packed
multi-question forward. These deterministic token inputs are numerical tests,
not a held-out natural-language quality or calibration cohort.

## Comparisons

| Backend | Maximum base-logit error | Changed-candidate error | Sampled hidden error |
|---|---:|---:|---:|
| SIMD | 1.4007092e-6 | 4.6472996e-7 | 5.7220459e-5 |
| NVIDIA | 1.3411045e-6 | 4.8056245e-7 | 6.8664551e-5 |

Gates remain `3e-4` logits and `2e-3` hidden. Each backend's256/512-capacity
results are bit-identical despite different group boundaries: capacity256 fits
two candidates/group, while512 fits six. Hidden comparisons execute the actual
packed group containing each selected candidate and remap local sample offsets.
They do not merely rerun the selected candidate alone.

Two goroutines begin together on one scorer: one reverses all candidates, the
other changes only the final candidate's first token. The scorer serialises
scratch use. Reversed logits remap exactly; all63 unchanged sibling scores stay
exact; the changed score matches its independent expected value. Earlier
returned output stays unchanged throughout. Both capacities pass ordinary tests;
capacity512 passes this concurrent test under the race detector.

## Resources and scope

Runs used one checkpoint/accelerator per process on i7-12700, RTX3060,
driver580.173.02, Go1.26.3/Linux amd64 and `GOMAXPROCS=6`. Maximum resident set:

| Backend / capacity | Ordinary peak RSS KiB | Race peak RSS KiB |
|---|---:|---:|
| SIMD /256 | 6,099,172 | not run here |
| SIMD /512 | 6,089,144 | 15,340,840 |
| NVIDIA /256 | 6,079,816 | not run here |
| NVIDIA /512 | 6,138,288 | 17,988,352 |

Post-GC live heap after concurrency was below the warmed baseline by64–84KiB,
with scorer, CPU weights and outputs explicitly kept live. A32MiB regression
allowance remains in the test; it is not observed growth. Whole-test wall times
include loading, hash checks, hidden probes and GC: ordinary SIMD141/105s,
NVIDIA19/16s; race SIMD499s, NVIDIA85s. These are not latency benchmarks.

The previous full three-round SIMD retention race timeout is still open. This
new pass covers two concurrent maximum-total requests, not eight concurrent
maximum-total requests, arbitrary queued callers or prolonged retention.

## Gates and reproduction

Focused review found no issue in oracle independence, grouped sample mapping,
input ownership or concurrency assertions. The default whole-tree CPU race
suite exits0; vet/build and ARM64/RISC-V test-binary cross-builds pass. Foreign
binaries were not executed. Default tests remain offline and small; released
execution is explicitly gated.

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/mojev_oracle_grouped_text.py UPSTREAM CHECKPOINT OUTPUT.json

GOMAXPROCS=6 GO_PHERENCE_MOJEV_GROUPED_BACKEND=nvidia \
  GO_PHERENCE_MOJEV_GROUPED_CAPACITY=512 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestReleasedGroupedTextScorer$' -v -count=1 -timeout=240s
# SIMD: GROUPED_BACKEND=simd, GO_PHERENCE_DISABLE_NVIDIA=1, timeout600s.
# Ordinary runs were also made at capacities256 and512 in separate processes.
```

`GO_PHERENCE_MOJEV_GROUPED_REPORT` optionally writes results and heap snapshots.
Evidence is under `/workspace/tmp/mojev-grouped-parity-20260926`.

This closes independent logits and sampled group-hidden parity for the tested
4096-total request. Other token distributions, every hidden row, natural-language
quality, calibration, prolonged admission and native foreign execution remain
unqualified. `RuntimeReady=false`.
