# Needle allocation audit, second pass -- 2026-09-20

The second allocation pass covers both Needle generations and more than steady cached decode. It measures FP32 and source CQ inference, cached prefix ingestion, loss/backward, LoRA gradients, Needle 2 contrastive/confidence heads, Needle 3 embedding/confidence/router heads, and the released Needle 3 archive. The [first audit](needle-allocation-pprof-audit-20260920.md) remains the historical baseline for the earlier cache/packed work.

The result is mixed rather than a blanket “allocation problem solved” claim: inference and head allocation counts fell by roughly **85–90%**, cached-prefix bytes by roughly **94%**, and training allocation counts by roughly **24%**. Warm released-model decoded execution is down to about **12.9k allocations and 8.82 MB per prompt**, but that remains too high for a zero-allocation decoder target. Packed execution remains slower and allocates more.

## Workloads and method

The source matrix uses the pinned tiny Needle 2 and Needle 3 fixtures, including each generation's CQ/A8 contract. Each benchmark reports one full five-token operation:

* full-prefix `Forward`;
* a decoder reset plus all prefix `Step` calls;
* loss plus all trunk gradients;
* loss plus all LoRA gradients;
* each supported auxiliary head.

Before/after results use identical code fixtures and `GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2`, with five baseline samples and ten final samples at `-benchtime=200ms`. Medians are reported; timing on a shared host is indicative, not a hard gate. Allocation counts and bytes are stable within each configuration. Exhaustive `-memprofilerate=1` profiles were collected separately from timing, with `alloc_objects`, `alloc_space`, `inuse_objects` and `inuse_space` views.

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 go test ./model/needle \
  -run '^$' -bench '^BenchmarkNeedleAllocationMatrix$' \
  -benchmem -count=10 -benchtime=200ms
```

## Source-fixture results

| Workload | ns/op before -> after | B/op before -> after | allocs/op before -> after |
|---|---:|---:|---:|
| Needle 2 FP32 forward | 65,028 -> 57,959 | 72,615 -> 78,712 | 800 -> 101 |
| Needle 2 CQ/A8 forward | 325,666 -> 399,272 | 80,776 -> 88,216 | 866 -> 130 |
| Needle 2 FP32 cached prefix | 82,012 -> 78,147 | 148,686 -> 8,932 | 577 -> 335 |
| Needle 2 CQ/A8 cached prefix | 85,051 -> 75,182 | 160,976 -> 8,933 | 580 -> 335 |
| Needle 3 FP32 forward | 81,811 -> 68,010 | 107,051 -> 119,116 | 1,172 -> 126 |
| Needle 3 CQ/A8/KV8 forward | 375,526 -> 413,634 | 116,939 -> 128,236 | 1,262 -> 154 |
| Needle 3 FP32 cached prefix | 106,322 -> 92,192 | 190,898 -> 11,018 | 760 -> 405 |
| Needle 3 CQ/A8/KV8 cached prefix | 108,978 -> 99,735 | 202,419 -> 11,019 | 765 -> 405 |
| Needle 2 contrastive head | 75,393 -> 56,056 | 78,840 -> 82,952 | 859 -> 110 |
| Needle 2 confidence head | 76,211 -> 57,664 | 79,937 -> 87,472 | 864 -> 114 |
| Needle 3 embedding head | 178,349 -> 150,755 | 239,716 -> 246,084 | 2,556 -> 258 |
| Needle 3 confidence head | 174,994 -> 149,933 | 239,667 -> 246,115 | 2,560 -> 259 |
| Needle 3 router head | 179,184 -> 152,387 | 239,802 -> 246,091 | 2,560 -> 259 |

Uncached inference/head slabs trade 2.6--11.3% more `B/op` for 85--90% fewer objects. The arena uses fixed 1,024-float, 1,024-int, 64-value and 128-pointer blocks; it intentionally does not pool across independent requests. The slack is bounded by the per-request workspace limit and shown here rather than hidden. CQ timing varied enough that the median regressed 10--23% in this short shared-host run; no CQ speedup is claimed. FP32 and head medians improved by roughly 11--26%.

Cached execution retains and clears arena blocks inside its decoder session. That removes repeated block allocation without sharing mutable scratch between decoders: cached bytes fell 94%, while allocations fell 42% for Needle 2 and 47% for Needle 3. `Reset` preserves bounded capacity and clears used float, integer, pointer and descriptor storage. A regression test requires zero heap allocations for an identical warm arena reset/reuse cycle and verifies no stale values or growth.

## Training results

| Workload | ns/op before -> after | B/op before -> after | allocs/op before -> after |
|---|---:|---:|---:|
| Needle 2 FP32 loss/all-grad | 203,878 -> 184,605 | 301,319 -> 296,776 | 5,234 -> 3,953 |
| Needle 2 FP32 LoRA-grad | 238,752 -> 230,890 | 331,788 -> 327,244 | 5,403 -> 4,122 |
| Needle 2 CQ/A8 loss/all-grad | 470,235 -> 484,231 | 315,914 -> 311,369 | 5,340 -> 4,039 |
| Needle 2 CQ/A8 LoRA-grad | 509,392 -> 595,622 | 346,380 -> 341,836 | 5,509 -> 4,208 |
| Needle 3 FP32 loss/all-grad | 258,409 -> 224,876 | 382,733 -> 377,230 | 6,084 -> 4,598 |
| Needle 3 FP32 LoRA-grad | 316,616 -> 263,113 | 415,469 -> 409,963 | 6,301 -> 4,815 |
| Needle 3 CQ/A8/KV8 loss/all-grad | 567,260 -> 492,867 | 400,300 -> 394,796 | 6,238 -> 4,720 |
| Needle 3 CQ/A8/KV8 LoRA-grad | 571,127 -> 555,128 | 433,039 -> 427,533 | 6,455 -> 4,937 |

Each differentiable value previously allocated separate backing arrays for values and gradients. One `2*n` array now provides two disjoint, capacity-limited slices. This removes one object per value while preserving lifetime and gradient isolation. A regression test fixes the warm primitive at two heap allocations (descriptor plus combined storage) and verifies that writes do not alias.

Pre-sizing per-execution parameter maps removes another small set of map-growth allocations from training. The final exhaustive Needle 3 FP32 loss profile still attributes most objects to tape values and closure-based reverse operations. Replacing closures with a typed operation tape is a larger design change and remains open. Training's remaining 4.0--4.9k allocations per tiny operation are not presented as satisfactory final numbers.

## Released Needle 3

The opt-in released archive benchmark loads once, prepares a decoder once, then measures reset plus a fixed chat prefix. Ten one-iteration samples on the Intel i7-12700 gave:

| Path | Time range | B/op | allocs/op |
|---|---:|---:|---:|
| Decoded | 172.8--200.5 ms | 8,820,880 | 12,933 |
| Hybrid packed | 281.4--454.8 ms | 12,903,056 | 14,263 |

The first published audit reported approximately 94 MB and 36.3k allocations for the same released decoded benchmark after its earlier changes (and roughly 120 MB before them). The current result is substantially lower, although changes between the audit points include more than this second pass. It is not a controlled claim that this patch alone removed the entire difference. Packed projections remain opt-in and slower.

A `-memprofilerate=1` released profile includes one-time archive parsing/materialization because Go benchmark profiles cover setup as well as timed iterations. Allocation-object views show per-step cached attention, trunk closures and dynamic names among the hot paths; allocation-space views are dominated by cold CQ decode and transposition. Those cold costs are intentionally separated from `B/op`, which covers the timed warm decoder loop.

## Implementation and ownership

* Ordinary inference uses immutable model tensor slices directly when weights do not require source CQ preparation; returned logits remain owned arena storage that escapes safely with the request. Source CQ weights are quantized once per request rather than once per layer reference.
* Decoder sessions retain request-local float, integer, tensor-descriptor and pointer blocks. There is no global pool and no mutable scratch shared between concurrent calls. Transactional cache publication remains after successful finite output and cancellation checks.
* The arena uses fixed-size blocks instead of geometric growth. A geometric version cut object counts but increased tiny forward bytes by 8--33 KiB and was rejected. A training descriptor arena increased bytes by roughly 93--111 KiB and failed to reduce allocations; it was also rejected.
* Training values combine `x` and `g` storage but retain distinct slices. Optimizer/model ownership and full-gradient maps are unchanged.
* Numerical kernels, quantization rules and tolerances are unchanged. Existing FP32/CQ logits, loss, all-gradient, cached/full-prefix, head, archive and native tests pass.

## Coverage and gates

`go test ./model/needle -coverprofile=...` reports **85.1% statement coverage** for the package as a whole. This is below the new 90% changed-package target; the new arena, allocation-regression and cross-generation benchmark paths are covered, but a scoped changed-line report is not available from the standard tool. The shortfall remains open rather than being labelled compliant.

Final native ARM64 validation on the CIX P1 used `GOMAXPROCS=2 nice -n 10` and passed **122 model, 58 loader, 11 CLI and 565 SIMD tests/subtests**. Three SCP attempts through the `.local` hostname failed with `Connection closed`; retrying the same bundle through its resolved LAN address completed normally. Whole-tree race passed for 114 packages (54 without tests); vet/build, CPU-feature-disabled affected tests, RISC-V cross-build, docs and frozen-artifact checks passed. No GPU use, health probe, service change, evaluation resumption or tolerance widening is part of this pass. Independent review timed out and provides no coverage.

## Remaining allocation work

The released decoder is not zero-allocation. Dynamic layer/engram names, attention score/tensor setup, output ownership and reset bookkeeping remain measurable. Training still creates thousands of operation closures and tensor buffers per tiny step. A typed backward-op representation, planned gradient storage, prepared integer indices and more persistent decoder workspace deserve separate measured changes; they must preserve concurrent model use, transactional rollback and gradients. Packed execution still needs both allocation and arithmetic work before it can be recommended for performance.
