# MoJev direct token segments

Native text scoring no longer materialises packed IDs and boolean masks only to
scan them back into token segments. It passes the validated encoded row directly
to inference and counts its tokens for usage. The public packed-row APIs remain
available and produce the same masks, ordering and owned outputs.

## Change

`encodeTextRows` extracts the existing encoding stage from `PackTextRows`.
It preserves prompt-first encoding, state/question truncation, untruncated
candidates, nonempty/nonnegative segment checks, per-segment 4096-token bounds,
and the 4096-total-token row limit. Both paths call that stage; packed callers
then use `PackEncodedRows`. Direct inference uses one encoded row and
`encodedRowTokenCount`. Answer assembly shares `assembleTokens`; the packed
wrapper derives the same count from the packed mask.

No encoder/head arithmetic, numerical tolerance or cancellation boundary changes.
All results remain caller-owned. Encoding scratch is request-local; truncated
token slices may retain their bounded original segment backing arrays until
scoring ends. No cross-request cache or shared mutable scratch was introduced.
Future additions to `PackEncodedRows` validation must also be considered at the
shared encoding boundary; focused equivalence tests cover current constraints.

## Measurements

Baseline is `060b62c0`, including the tokenizer allocation improvement. Host:
Linux amd64, Go 1.26.3, i7-12700, `GOMAXPROCS=6`; RTX 3060/driver580.173.02 for
NVIDIA. Approved checkpoint/tokenizer hashes are checked before loading by the
existing four-workload probes. Weight SHA-256:
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
Only one checkpoint/backend process runs at a time.

A synthetic two-field preparation benchmark uses fresh token slices from an
injected encoder, state/question truncation and seven candidates. The baseline
packs then extracts spans (with exact slice capacity); direct mode retains the
validated segments. Ten samples per mode at `-benchtime=100ms`, setup excluded:

| Per operation | Packed/extracted | Direct | Change |
|---|---:|---:|---:|
| Time | 14.342 µs | 2.717 µs | −81.06% |
| Bytes | 33.15 KiB | 11.14 KiB | −66.39% |
| Allocations | 72 | 27 | −62.50% |

All benchstat comparisons have p<0.001, n=10. This measures preparation only,
not tokenisation or inference.

One warm-up and five full calls per real workload yielded these median request
allocation counts, comparing the prior tokenizer-only probe with direct mode:

| Workload | GPU before → direct | SIMD before → direct |
|---|---:|---:|
| Short, two choices | 381 → 350 | 368 → 349 |
| Short, eight choices | 588 → 540 | 579 → 530 |
| Two questions, two choices | 618 → 572 | 585 → 544 |
| Longer, two choices | 585 → 552 | 571 → 537 |

Two additional interleaved SIMD pairs confirmed lower counts: direct ranges
343–348 / 525–529 / 541–546 / 536–538 versus baseline
373–375 / 574–583 / 586–588 / 571. Responses remain exact, including probability
maps and usage. GPU medians remain roughly 37/40/72/70 ms. SIMD timings vary;
several direct runs are slower. No full-inference speedup is established.

The first SIMD workload showed roughly 32–35 KB more allocated bytes in direct
runs despite fewer allocation objects. The cumulative allocation-profile diff
attributes 82 KB less allocation to `scoreTextContextWith` across the profiled
requests, and removal of 38.77 KB attributed to `PackEncodedRows`. It did not
identify the first-workload spike. Full-request bytes remain unqualified for
that workload; the accepted benefit is fewer objects and lower isolated
preparation bytes. No peak/retained-heap improvement is claimed.

## Verification

- Direct segments match packed-span extraction and explicit expected token IDs
  for multiple fields, unequal candidate counts and state/question truncation.
- Candidates remain untruncated. Total4096 is accepted; total4097 is rejected.
  Empty/oversized segments, negative IDs even beyond the truncation boundary,
  malformed logits and callback failures return no partial output.
- Direct answers/usage match the existing packed public assembler; repeated
  responses remain independently owned. Existing cancellation boundary tests pass.
- Pinned tokenizer/packing/response/control tests pass under race three times.
- Real GPU and SIMD released inference tests pass under race, serially. Maximum
  sampled hidden errors are `7.34329e-5` / `9.91821e-5`; existing gates remain
  `3e-4` logits and `2e-3` hidden. No fixtures were regenerated.
- Whole-tree NVIDIA-disabled race exits zero; vet/build, docs/layout and Linux
  ARM64/RISC-V cross-builds pass. Foreign binaries were not executed.

Default changed-function coverage: `encodeTextRows`95.8%,
`encodedRowTokenCount`100%, `PackTextRows`100%, packed `assemble`100%,
`assembleTokens`91.7% and `scoreTextContextWith`100%. Focused read-only review
found no current semantic/ownership issue and highlighted keeping validation
aligned with the packed path. A delegated test draft needed literal-newline
syntax corrections before compiling; no failed draft was published.

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./model/mojev -count=10
go test ./model/mojev -run '^$' -bench '^BenchmarkDirectSegments' \
  -benchmem -benchtime=100ms -count=10

GOMAXPROCS=6 GO_PHERENCE_MOJEV_NVIDIA=1 \
  GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/dev/shm/mojev-checkpoint \
  go test -race ./model/mojev -run '^TestMoJevAcceleratedReleased$' \
  -v -count=1 -timeout=240s
# SIMD: disable NVIDIA, set GO_PHERENCE_MOJEV_SIMD=1; timeout480s.

GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s
```

Raw samples, profiles, comparisons and test logs are under
`/workspace/tmp/mojev-direct-segments-20260926`. Shared-loader tokenizer changes
are documented separately in [the tokenizer report](mojev-tokenizer-allocations-20260926.md).
Native foreign execution, held-out quality/calibration and hours-long retention
remain open. `RuntimeReady=false`.
