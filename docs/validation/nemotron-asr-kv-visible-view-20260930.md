# Nemotron ASR immutable K/V visible views — 30 September 2026

A private immutable K/V view removes 313.8 MB of allocation per 11-second JFK ASR request on Sigma, a 24.4% reduction from the deployed parallel-dense baseline. The measured latency reduction is small; no new large speedup is established. Four other CPU candidates were rejected.

## Ownership and cache transition

`Encoder0Attention.ForwardCachedChunk` now calls a private `Encoder0KVCache.updateVisibleView` method. The attention operator's visible K/V arrays never escape that operator. Instead of copying retained tails into another pair of allocations, the prepared cache retains an immutable head-strided view into those visible arrays. `stride` describes the head extent; `skip` excludes the leading rows dropped by the released sliding-window rule.

The next update allocates fresh visible buffers and copies only the retained rows. No previously committed backing array is mutated. The existing block/tower transaction still prepares state copies and commits after successful finite output. For the usual four-row streaming chunk, at most 60 rows remain allocated per layer; only the latest 56 are retained history. Larger legal standalone inputs remain bounded by retained plus current rows.

Public `Update` continues to return independently owned visible buffers and retain separate packed history. Public `Snapshot` packs retained rows into independent copies. Mixing private views with a later public update is tested. This change does not alter numerical operations, model weights, prompt selection, word timing, speaker attribution or cancellation semantics.

## Matched local evidence

Source baseline: `9b1e8380c952fa1610fed42cccb86ed0002895cb`, with the previously deployed four-worker dense optimisation. Go 1.26.2, i5-1340P, four Go threads and four CPU quota. Both executables use the pinned ASR checkpoint/tokenizer from [the replacement record](nemotron-asr-sigma-trial-20260930.md).

Tests ran in a network-disabled container with 8 GiB cap and no container swap or GPU devices. A once-per-second guard stopped only the experiment if the live Qwen slot became active/unobservable or host available memory fell below 6 GiB. Native trial windows were explicitly coordinated; neither live service was changed during comparisons.

JFK whole-request benchmarks excluded model/WAV load. Interleaved baseline/view samples each used two requests:

| Measurement | Baseline sample 1 | Baseline sample 2 | View sample 1 | View sample 2 |
| --- | ---: | ---: | ---: | ---: |
| Time per request | 4.4796 s | 4.4600 s | 4.3741 s | 4.3837 s |
| Allocated bytes/request | 1,285,962,632 | 1,285,957,832 | 972,171,688 | 972,174,520 |
| Allocations/request | 107,404 | 107,394 | 105,715 | 105,720 |

Mean allocated-byte reduction: **313,787,128 bytes/request (24.4%)**. Mean time changed from 4.4698 to 4.3789 seconds (2.0%). Two samples cannot establish a sustained latency gain. This is cumulative allocation, not process RSS, persistent model size or an asserted 314 MB peak-memory decrease.

Separate recorded-output runs compared every token ID, absolute encoder frame and text value against the baseline. All were unchanged on JFK (185 decisions), the 20-second podcast crop (419 decisions), and a fresh repeated JFK stream. Timings for these single runs were 4.705→4.609 s, 7.872→7.757 s and 3.835→3.990 s respectively; the slower repeated-JFK sample reinforces the limited latency claim.

## Rejected experiments

Only the K/V-view change remains in the release. No rejected SIMD/pool code is installed or committed.

| Candidate | Observation | Decision |
| --- | --- | --- |
| Three persistent dense workers plus caller | Allocation count fell from about 107k to 31k/request; baseline 7.250/6.940 s and pool 6.972/6.943 s overlap | Reverted: no convincing whole-request speed gain; avoid new process-lifetime worker machinery |
| Two-row AVX2 weight sharing, with original per-dot arithmetic | Bit-exact matrix tests passed; combined cache+kernel 4.382/4.400 s versus baseline 4.437/4.425 s did not beat cache-only results | Reverted: no demonstrated incremental gain |
| Contiguous worker output ranges | Combined cache+dispatch 4.369/4.380 s versus baseline 4.485/4.427 s was effectively cache-only performance | Reverted: no demonstrated incremental gain |
| Four-row AVX2 weight sharing | Bit-exact tests passed, but combined candidate 5.404/5.464 s versus baseline 4.381/4.409 s was about 24% slower | Reverted: repeatable regression |

Absolute timings across different batches were variable. The table uses each candidate's paired baseline; it does not compare an unusually slow early baseline against later candidates to claim a gain. No GPU experiment or precision change was made.

## Verification

- Public/private view equivalence, packed snapshots, mixed public updates, immutable transaction rollback, malformed/nonfinite rejection and bounded storage tests passed.
- Recorded native baseline/view tokens, frames and text matched exactly for all three cases.
- Released shared-model independent-stream test passed for 4,040- and 80,000-sample PCM calls; SIMD-only cancellation-after-chunk test passed.
- Full `go test -p 1 ./...`, `go vet -p 1 ./...` and `go build -p 1 ./...` passed with NVIDIA disabled and CGO disabled after reverting the rejected candidates.
- `git diff --check` passed. Numerical tolerances were unchanged.
- A race-detector run was unavailable because no C compiler is installed; concurrency tests do not constitute race-detector evidence.

This is implementation agreement against the previously deployed native model, not independent PyTorch or labelled multi-language accuracy qualification.

## Release and remaining work

The cache-only change is ready for an idle-queue update. Live latency/peak memory must be reported separately after installation. Profiles, allocation counts, native outputs and rejected-candidate logs are retained under `tmp/nemotron-asr-opt2-20260930/`; local model assets/audio and experimental source copies are excluded from Git.

Future CPU work should target the remaining measured arithmetic or copying cost without trading exact output/rollback guarantees for an unmeasured speed claim. No GPU clearance, model change or Qwen runtime change is part of this release.
