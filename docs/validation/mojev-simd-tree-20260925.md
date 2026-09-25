# MoJev SIMD shared ancestors

The CPU SIMD scorer now evaluates a question's state and question rows once,
then forks candidate recurrence inside the same bounded tree forward. It uses
the existing assembly projections and attention operations. The NVIDIA path
and CPU reference still execute separate candidate branches.

## Tree execution

A tree consists of contiguous state rows, question rows and 2–64 candidate
segments. Each segment is nonempty. Projection/normalisation/residual operations
batch all tree rows. At each linear layer, convolution walks explicit parent
indices, skipping siblings. The recurrence saves one SSM snapshot after the
question and restores it at each subsequent candidate boundary. No recurrent
state survives the forward.

Full attention sees state only for state rows, state plus question for question
rows, and those ancestors plus the current candidate for candidate rows. Each
candidate uses RoPE positions beginning at `stateLen+questionLen`, independent
of its packed offset. Attention reductions retain the same ancestor/candidate
order as separate execution.

Trees reuse the scorer's existing token capacity, at most 512. If any question's
whole tree does not fit, the request falls back to separate candidate execution;
each branch must still fit. Validation checks all IDs and lengths before model
work. Output remains owned, and failures return no partial logits. Questions
are processed separately, so cross-question sharing is not implemented.

Extra permanent scratch is one `16×128×128` F32 SSM snapshot (1 MiB). Tree-index
arrays and head masks are bounded. No cross-request cache, retained caller
slices, wider numerical tolerance or precision change was introduced.

## Paired measurements

The approved pinned checkpoint, i7-12700, Go 1.26.3, Linux/amd64 and
`GOMAXPROCS=6` are unchanged. Checkpoint revision
`0c8695b6252f4205907433d4e196a94f032e60c3`, SHA-256
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
The preserved `3860f490` SIMD binary is the before baseline; subsequent published
commits changed only NVIDIA launch bindings. Processes run sequentially, one
warmup plus five measured full-library requests per workload. JSON decode,
tokenisation, inference, response assembly and marshal are timed. Preparation,
hash checking and HTTP are excluded.

| Request | Separate ms | Tree ms | Allocations separate → tree | Bytes separate → tree |
|---|---:|---:|---:|---:|
| Two choices | 458.15 | 245.04 | 694 → 660 | 93,248 → 41,016 |
| Eight choices | 2561.25 | 423.03 | 1,180 → 999 | 180,488 → 52,560 |
| Two questions | 891.51 | 482.38 | 1,055 → 999 | 101,296 → 65,256 |
| Longer context | 1552.91 | 802.31 | 1,620 → 1,597 | 96,592 → 74,272 |

Medians improve by 46–84%; the eight-choice case avoids seven repeated ancestor
passes. An earlier independent tree run measured 247.57/349.05/458.74/741.67 ms,
so exact latency remains host-sensitive. Saved public responses match the
separate-branch baseline exactly. These are execution benchmarks, not held-out
task-accuracy results; no significance test is reported.

Ordinary sampled RSS at the final workload was 6,059,608 KiB before and
6,082,940 KiB after. This is sampled resident memory, not a peak or long-term
retention guarantee. The extra logical SSM budget is 1 MiB; allocator/runtime
variation also contributes to process RSS. Weight packing still dominates host
memory. The post-change profile assigns 76.17% of CPU samples to the assembly
matrix microkernel, with less total repeated work.

## Verification

- Eight independent released F32 cases: maximum logits error `4.6759844e-5`
  against `3e-4`; hidden reference gate `2e-3` remains unchanged.
- Exact tree-versus-separate logits for 2, 8 and 64 unequal-length candidates,
  reverse permutations and a tree exceeding scratch capacity that takes the
  branch fallback. Existing substitution/length/question/order tests pass.
- Released race test covers repeatability, four callers and owned outputs.
- Synthetic attention tests poison sibling K/V rows and compare to physically
  compacted ancestors/candidate rows, with native and disabled dot dispatch.
  Warm tree attention allocates zero with caller buffers.
- Boundary tests reject empty/overlapping/unsorted/out-of-range candidates and
  preserve destination sentinels on failure.
- Whole-tree CPU race suite exits 0; focused races, vet, build and ARM64/RISC-V
  cross-builds pass. Native non-amd64 execution is unqualified.

The focused independent review timed out; it did not produce approval or
findings. Final independent review and broader changed-code coverage remain
open. NVIDIA shared ancestors, cancellation, maximum-capacity admission,
held-out calibration/quality and service qualification are separate work;
`RuntimeReady` stays false.

Commands and samples are under `/workspace/tmp/mojev-tree-20260925`. The
released test is `GO_PHERENCE_MOJEV_SIMD=1` with the same hash-checked
`GO_PHERENCE_MOJEV_CHECKPOINT_DIR` setup used by the
[acceleration tests](mojev-accelerated-text-20260925.md).
