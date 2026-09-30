# Nemotron ASR small-query parallel dense dispatch — 30 September 2026

Parallel output-channel tiling reduced native CPU ASR request time on Sigma while preserving all recorded token decisions, encoder frames and text. The change reuses the existing blocked GEMM arithmetic; it introduces no model, precision or GPU change.

## Dispatch and numerical contract

`DenseNTTo` now uses the existing `sgemmNTBlockedParallelTo` for contiguous blocked-kernel shapes with `2 <= M <= 5`, `N >= 1024`, `K >= 1024`, `alpha == 1` and more than one Go worker. It caps this path at four workers. Larger-M and other-alpha/stride paths retain their existing dispatch.

The helper partitions N into 64-output-column tiles. Each worker writes distinct output elements and runs the original `SgemmNTBlockedFMA` K-block sequence. The accumulating destination contract is unchanged, including nonzero initial C. It does not combine independent time chunks or alter the native four-row streaming/cache schedule. Submitted SIMD tiles finish before callers return; kernels are not interruptible.

New tests compare float32 bits with the serial blocked implementation for FFN/QKV-style shapes, nonzero destinations, odd/tail dimensions and independent concurrent calls. The original numerical tolerances were not changed. This is generic checked dense dispatch, not a model-name special case.

## Baseline attribution

Baseline source: `db6bdbbf1cf6db7698398a92384e42b5ef29a789`, Go 1.26.2, i5-1340P, four Go threads. The native ASR checkpoint/tokenizer are pinned in [the service replacement record](nemotron-asr-sigma-trial-20260930.md).

A one-request CPU profile sampled the blocked SIMD tile at 60.8% flat CPU and the cached tower at 70.1% cumulative CPU. The profile includes model loading, where `GetFloat32` accounts for 13.1% flat CPU; these shares must not be treated as exact inference-only fractions. The allocation profile likewise includes loading (2.38 GB in `GetFloat32`) and identifies K/V-cache update allocation as the next substantial request cost.

## Matched native measurements

Trials ran in a network-disabled, GPU-device-free container: four CPU quota, four Go threads, 8 GiB cap and no container swap. A live guard checked Qwen's slot and host available memory every second, stopping only the experiment if Qwen became active/unobservable or host available memory fell below 6 GiB. Live Qwen and the transcription service remained unchanged during comparisons.

The baseline and candidate test executables used identical source apart from `dense.go`. Whole-request JFK benchmarks exclude loading and use two requests per sample in interleaved baseline/candidate order. All samples returned the expected 185 decisions.

| Arm | First paired sample | Second paired sample |
| --- | ---: | ---: |
| Serial blocked baseline | 6.536 s/request | 6.543 s/request |
| Four-worker parallel | 4.676 s/request | 4.415 s/request |

Mean of the two sample means: 6.540 s versus 4.545 s, **30.5% less processing time** (1.44× throughput). These are short samples, not a sustained/corpus benchmark. The four-worker dispatch adds about 5.38 MB/request (0.42%) and about 85,000 small allocations from goroutine/work distribution: baseline 1.281 GB/22,344 allocations, candidate 1.286 GB/about 107,400 allocations. A two-worker sample measured 4.879 s/request and about 73,400 allocations; it was slower than four workers here. A persistent worker pool may reduce allocations, but was not added without independent lifecycle and speed evidence.

Separate recorded-output runs used fresh streams and compared every token/frame/text value:

| Clip | Baseline | Candidate | Time reduction |
| --- | ---: | ---: | ---: |
| JFK, first request | 6.810 s | 4.661 s | 31.6% |
| Podcast, 20 s crop | 11.850 s | 7.866 s | 33.6% |
| JFK, fresh repeated stream | 6.127 s | 3.909 s | 36.2% |

JFK had 185 decisions and podcast 419 in both arms. All decision IDs, absolute encoder frames and decoded text matched exactly. This is baseline/candidate implementation parity; it does not add independent PyTorch or human-labelled accuracy qualification.

## Verification

- Full `go test -p 1 ./...`, `go vet -p 1 ./...`, and `go build -p 1 ./...` passed with NVIDIA disabled and CGO disabled.
- Dense blocked-bit exactness, independent simultaneous calls and Whisper attention exactness tests passed.
- `GODEBUG=cpu.all=off` and one-worker focused regressions passed.
- Opt-in released ASR shared-weight independent-stream test passed (4,040- and 80,000-sample PCM calls).
- Released CPU cancellation-after-chunk test passed: stream closed, no partial decisions returned.
- Race testing was not available because the host has no C compiler. Concurrent tests above are not a race-detector claim.
- One invocation accidentally selected GPU subtests along with the CPU cancellation test. PTX and Vulkan failed availability checks in the isolated no-device container; no GPU kernel executed. The explicitly selected SIMD-only subtest then passed. No GPU execution/recovery is authorised.
- `git diff --check` passed.

## Evidence and release state

Local comparison/profiles remain under `tmp/nemotron-asr-opt-20260930/`: `matched-*.log`, `baseline-results.json`, `parallel-results.json`, CPU/allocation profiles, native shared-state/cancellation logs and memory guard output. Model assets/audio are not committed. Full logs: `/tmp/nemotron-opt-full-{tests,vet,build}.log`.

Implementation commit `51f546d604cf6744d25ee8c3244827c6ed80abfc` was pushed and remotely verified. The authorised idle transcription service was updated to server SHA256 `906e0700670c2684c5334c52a756676271ee62509326cb8a15474ac3c79a01cd`. Frontend/model/profile settings were unchanged; the runtime SHA intentionally changes plan identities. Old checkpoints are not accepted as new-plan output.

Before the switch, all eleven existing manifest hashes were recorded and matched after binary/config installation. Backup: `deploy-backup-nemotron-opt-20260930T140739Z`. Metadata-only check passed with no model loaded or listener. The service retains four CPU quota, 8 GiB limit, no swap and no GPU devices. Live Qwen PID/restart count remained unchanged; Gemma remained off.

| Live completed job | Before, retained sample | After | Time reduction |
| --- | ---: | ---: | ---: |
| JFK ASR, 11 s | 6.870 s | 5.013 s | 27.0% |
| Podcast ASR, 20 s | 11.595 s | 8.365 s | 27.9% |
| JFK ASR plus speakers, 11 s | 8.942 s | 6.168 s | 31.0% |

These are persisted `updated - created` complete-job times, including decode/queue/export publication, excluding service startup and verifier cleanup. Before/after live samples are descriptive; the interleaved native results above are the primary causal comparison. After RT factors are 0.456, 0.418 and 0.561 respectively. All jobs completed in one attempt; text was unchanged and all artifact download sizes/SHA256 matched. Earlier retained exports remained downloadable after the update. The queue drained to its original terminal failed item. The service had zero restarts and zero cgroup swap; peak cgroup memory was 6,911,991,808 bytes (about 6.44 GiB, including file-backed pages). Host available memory remained above its guard.

The CPU window was explicitly released back to the Qwen investigation after live checks. GPU acceleration, longer recordings, external-request contention, multilingual accuracy, incremental ASR retention and persistent worker pools remain separate work.
