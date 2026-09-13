# SincNet strict tail diagnosis — 13 September 2026

## Result

The remaining lowered-filter SincNet strict mismatch is caused by a shape-dependent PyTorch CPU `conv1d` tail/epilogue reduction path. It is not a Go indexing error, padding error, filter-generation error, max-pool winner change, or standalone normalization error.

The unchanged strict gate reproduces 12 passing subtests and two failures (`wave-stride10-mode2` and `mode3`). Both fail at boundary index 1717 with maximum absolute error `0.00038086623` (`0.085615553` versus `0.0852346867`, gate `2e-4`). Scalar and SIMD FMA outputs remain bit-identical.

## Local amplification

Index 1717 is channel 39, pooled frame 40. The three raw stage-0 values feeding that pool are:

| Path | frame 120 | frame 121 | frame 122 | winner |
|---|---:|---:|---:|---:|
| PyTorch trace | -0.00006200524 | -0.0014827227 | -0.0012532701 | 121 |
| Go serial FMA | -0.00006184202 | -0.0014844541 | -0.0012535771 | 121 |

The max-pool winner does not change. The pooled delta is only `1.7314451e-6`. Applying the already-qualified exact normalization to the reference pooled row reproduces the reference boundary bit-exactly. Replacing only pooled frame 40 with the Go value produces boundary error `0.00038284063`, explaining essentially the complete observed `0.00038086623` mismatch. Frame 41 contributes only `-1.9744039e-6`; frame 42 contributes zero.

Thus whole-window instance normalization amplifies one local raw-convolution tail difference; normalization is not its source.

## Backend shape discriminator

The pinned environment is Torch `2.14.0+cpu`, one thread, MKLDNN disabled. Adding one through nine unused suffix samples to the exact 1531-sample normalized input leaves all 10,320 PyTorch outputs bit-identical. End alignment and ignored suffix contents are ruled out.

A prefix-only output-count sweep compares the same PyTorch `F.conv1d` with Go ascending-K float32 FMA using exact lowered filters:

| output frames | first/last differing frame | differing values |
|---:|---:|---:|
| 112–113 | none | 0 |
| 114–115 | 112–113 | 158 |
| 116–117 | 112–115 | 317 |
| 118–119 | 112–117 | 476 |
| 120–121 | none | 0 |
| 122–123 | 120–121 | 160 |
| 124–125 | 120–123 | 319 |
| 126–127 | 120–125 | 478 |
| 128–129 | 120–127 | 636 |

Every mismatch has the same maximum raw error, `2.288818359375e-5`. The onset resets at output counts divisible by eight and affects the final even-sized epilogue. This output-shape dependency identifies an internal blocked `conv1d` epilogue/reduction choice rather than convolution geometry or input data.

A materialized PyTorch `unfold` multiply/sum also differs from `F.conv1d`, confirming that the traced operator is not equivalent to a simple exposed tensor reduction whose order can be copied from source-level code.

## Decision

Do not widen the `2e-4` boundary tolerance, special-case silence, or add shape-specific arithmetic to production. Exact intermediate parity would require emulating undocumented PyTorch CPU backend dispatch and is neither portable nor justified by output quality: lowered-filter FMA endpoints, hard masks, complete diarization turns and retained DER gates already pass.

Keep the strict test as an opt-in diagnostic hold. Treat Community-1 implementation as endpoint-qualified but not bit-/tolerance-identical at every internal PyTorch boundary. This distinction must remain in documentation and qualification reports.

## Evidence

- `strict-current.log` — current 12-pass/2-fail strict gate.
- `tail-amplification.txt` — raw triplets, pool winners and one-row normalization amplification.
- `suffix-sensitivity.txt` — unchanged outputs for 0–9 ignored suffix samples plus `unfold` comparison.
- `output-sweep.txt` — output-count sweep proving the blocked tail pattern.

No trained model, GPU, service, deployment, default, tolerance or production arithmetic changed during these diagnostics.
