# MoJev long-input F32 normalisation correction

A new independent 512-token reference exposed incorrect epsilon placement in
repaired branch Q/K normalisation. The old Go/SIMD/PTX expression was
`1/(sqrt(sum(x²))+eps)`; pinned Transformers F32 recurrent attention uses
`1/sqrt(sum(x²)+eps)`. Correcting the branch paths reduces long-input hidden-row
error by more than an order of magnitude, with unchanged tolerances.

## Failure and correction

The first long reference run gave maximum sampled hidden error
`0.0020866394` on SIMD, above the existing `0.002` gate. NVIDIA gave
`0.0019874573`, narrowly inside it. Both had small logit error; head agreement
alone had hidden the drift. Replacing recurrent SIMD updates with the scalar
helper or reordering RMSNorm multiplication did not fix the failure, so those
experiments were reverted.

The repaired F32 branch now uses a separate L2 helper with epsilon inside the
square root. The ordinary causal/state API still selects its previous
normalisation and BF16 behaviour. The scalar branch, SIMD branch and MoJev PTX
kernel use the corrected formula. There is no tolerance widening, quantisation
or approximation mode. Scores can change slightly from previous versions.

## Independent long reference

`scripts/mojev_oracle_long_text.py` uses pinned upstream Python
`MoLeMo-Lab/mojev@a74d58cd19ec573e83e8e27f9fecd837b8d830fb`, Transformers5.17.0
and Torch2.14.0+cu130 on CPU. It checks the upstream model source, Transformers
implementation, config and weight hashes, upcasts existing BF16 values to F32,
and runs each candidate with fresh state and branch-local positions. No Go
outputs are used to create expected values.

The original eight-case fixture is unchanged. The additional fixture contains:

- State/question lengths128/128 and two256-token candidates:512-token paths.
- Two128-token candidates: a512-row shared tree, each separate path384 tokens.
- Sibling token substitution, sibling length change and candidate reversal.

Each case stores logits plus hidden rows sampled at the last state, question
and candidate token. Shared-tree hidden output is checked directly as well as
separate-branch output. The two offline final oracle runs were byte-identical.

Pins:

- Checkpoint revision `0c8695b6252f4205907433d4e196a94f032e60c3`.
- Safetensors SHA-256 `eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
- Generator SHA-256 `989bdc1d4ce4cfe7329ff23aed3285b7e350f00c976ee44ddf6ffa3da00c775d`.
- Fixture SHA-256 `5f1a5c6e891eb2bbc5c8de0ee3a87c6a41f28914302b649d136db8f71cd5a0bd`.

This validates the explicit repaired branch-local policy, not the leaky released
packed multi-question forward. It tests deterministic token inputs, not held-out
natural-language task quality or every hidden row.

## Results

| Backend | Long logits max error | Long sampled hidden max error |
|---|---:|---:|
| SIMD | 6.5565109e-7 | 9.1552734e-5 |
| NVIDIA | 9.5367432e-7 | 1.3351440e-4 |

Fixed gates remain `3e-4` logits and `2e-3` hidden. Sibling and permutation scores
remain exactly isolated. Both long-reference runs pass under the race detector,
separately: NVIDIA80s and SIMD319s including process/test overhead.

The original eight cases also improve. SIMD maximum logits/hidden errors are
`3.6954880e-6` / `9.9182129e-5`; NVIDIA `2.8312206e-6` / `7.3432922e-5`.
The CPU reference's original fixture passes with maximum logits `2.92063e-6`
and hidden `5.34058e-5`. Public causal/BF16 tests remain unchanged.

Near-zero synthetic vectors distinguish the formulas. Go branch-normalisation
helpers have100% scoped statement coverage. PTX tests also verify zero/tiny
vectors, ordinary values and preservation of the unnormalised V segment.
Memcheck reports zero errors; racecheck reports zero hazards. PTX regenerates
byte-for-byte. Combined accelerated short-reference races and whole-tree CPU
race pass; vet/build and ARM64/RISC-V test-binary cross-builds pass. Foreign native
execution is not qualified.

Post-fix five-sample full-request medians on i7-12700/RTX3060, Go1.26.3,
`GOMAXPROCS=6`, with one warmup and the unchanged four workloads:
SIMD236/369/508/757ms; GPU37.3/39.9/73.0/70.2ms. These confirm the previous
performance range, not a new speedup. Setup/hash checks are outside timing.

## CPU experiments rejected in this pass

No CPU projection runtime changes were retained. Five-sample trials of native
row-tile padding, padded input stride, software prefetch, four/eight-step
assembly unrolling and exact-order K-panel accumulation did not improve all
full-model workloads consistently. The K-panel prototype retained FMA order
and passed bitwise native/scalar differential tests, but its extra accumulator
loads/stores did not yield a dependable win.

Baseline medians across three runs ranged228–232/348–358/455–479/761–853ms.
The experiments ranged223–259/333–410/434–584/746–892ms. Individual wins were not
stable enough to justify extra code. All experimental source and binaries were
preserved outside the repository in `/workspace/tmp/mojev-cpu-tiles-20260926`;
tracked CPU projection files were restored before this correction.

## Reproduction and limits

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/mojev_oracle_long_text.py UPSTREAM CHECKPOINT OUTPUT.json

GOMAXPROCS=6 GO_PHERENCE_MOJEV_LONG_BACKEND=nvidia \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestReleasedLongTextScorer$' -v -count=1
# Run again in a separate process with LONG_BACKEND=simd and NVIDIA disabled.
```

The focused review confirmed the intended causal/BF16 scope and prompted direct
shared-tree hidden checks. The oracle remains dependent on pinned upstream
Python scorer/head semantics, independently implemented from the Go paths.
Evidence includes the original failing log and two final identical replays in
`/workspace/tmp/mojev-long-parity-20260926`.

Independent512-token branch/sampled-tree parity is now covered. Full4096-token
request accuracy, concurrent4096-token admission, prolonged retention, the full
three-round SIMD admission race, native foreign targets and held-out calibration
remain open. `RuntimeReady` stays false.
