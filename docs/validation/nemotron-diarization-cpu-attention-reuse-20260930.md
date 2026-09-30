# Nemotron diarization CPU attention-buffer reuse — 30 September 2026

`Layer1Attention.forwardOffline` now reuses its owned K projection for the output projection after attention has consumed K. It clears the buffer before the GEMM, preserving the accumulating-destination contract. The public attention and residual outputs remain independently owned and do not alias caller input or model weights. The change applies to all 30 indexed audio layers on the CPU SIMD path; it changes no numerical tolerance or GPU implementation.

The measured baseline was clean `7f85809b49103fcd80054564c727ccdf9431d7b9`, Go 1.26.3, Linux amd64, Intel i7-12700, `GOMAXPROCS=4`, released diarization checkpoint SHA-256 `c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6`. A one-request allocation profile of the 30-second JFK PCM streaming benchmark identified `Layer1Attention.forwardOffline` at about 952 MB of direct allocation (profile `alloc_space`) and `Layer1QKV.Project` at about 2,004 MB. Profiled runs use `-memprofilerate=1` for attribution, not timing. The latter cost and prepacked SGEMM CPU time remain after this change.

| 30-second JFK CPU PCM request | Before | After |
|---|---:|---:|
| `B/op`, normal benchmark | 6,003,754,704–6,003,756,550 (four completed five-iteration samples) | 5,490,772,928–5,490,774,672 (five two-iteration samples) |
| `allocs/op`, normal benchmark | 81,834–81,838 | 80,574–80,578 |
| Time per request | 10.96–12.22 s | 11.04–11.93 s |

The measured allocation change is about **513 MB and 1,260 allocations per 30-second request**, with no evidence of a timing regression or a robust timing improvement. The after-run values are from `GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL=... GOMAXPROCS=4 go test ./model/nemotrondiarization -run '^$' -bench '^BenchmarkPCMStreamingRequestThirtySeconds$' -benchtime=2x -count=5 -benchmem`. The before-run samples used `-benchtime=5x`, so timing ranges are descriptive, not a matched `benchstat` speed gate. Loading is outside the benchmark timer; the test loads the same released weights and repeats JFK PCM to 30 seconds.

The independently labelled AMI 530–630-second excerpt was run three times with a CPU-only binary built after this change. Load-inclusive process wall times were 76.14, 73.71 and 75.55 seconds, versus prior CPU runs of 84.85, 74.70, 75.08 and 75.56 seconds. All runs were under the 100-second input duration and their parsed 67 speaker spans had identical geometry. The timing ranges overlap; this is not a complete-request speedup claim. Exact hashes of saved outputs are for provenance, never a numerical acceptance gate.

Checks passed:

- Released-model layer-1 attention parity at 16 and 138 rows, complete layer parity, and offline tower parity. These compare error distributions against PyTorch fixtures under existing thresholds; no tolerance was widened.
- CPU PCM streaming logits/segments versus pinned PyTorch fixtures at 11, 31 (mixed speakers), and 100 seconds. The 100-second run had 9,999 logits rows, max absolute error `6.866455078125e-05`, mean absolute error `4.932524520705993e-06`, zero outliers under the original gate.
- New offline ownership test for scalar (1 row) and tiled (16 rows) paths: attention and residual outputs are independent of each other, caller input and subsequent calls.
- Full affected package and CLI tests, focused released-model race test, affected `go vet`, `go build ./...`, and Linux ARM64/RISC-V test-binary cross-builds. Cross-builds are not native execution.

Local evidence remains under `/workspace/tmp/nemotron-ami-es2004a-530-630/` (`diar-alloc.pprof`, `attention-reuse-alloc-single.pprof`, benchmark logs, parity logs and CPU CLI runs); no AMI media or large profiles are committed. The NVIDIA GPU became unavailable after a separate [Vulkan Xid79 failure](../../benchmarks/speech-foundations/ami-pilot-contract-20260913/nemotron-530-630-followup-20260930.md). No GPU inference or recovery was attempted for this change. GPU reliability, sustained multi-request service throughput, other-host performance and broader labelled-quality qualification remain open.
