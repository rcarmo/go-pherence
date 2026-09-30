# Nemotron ASR exact-order attention and convolution — 30 September 2026

Three CPU changes reduce Sigma's matched warm JFK request time from 4.1775 to 3.9661 seconds (5.1%) and remove about 55.1 MB of request allocation. The candidate preserves every recorded token, encoder frame and transcript on JFK, a fresh JFK repeat and a 20-second podcast crop. No precision, model, prompt, timing policy or GPU change is included.

## Changes

1. `AttentionValueRowTo` vectorises the cached-attention probability×value sum on amd64 with AVX2. Eight output dimensions are computed independently at a time. Each dimension visits source rows in the same order, with separate `VMULPS` and `VADDPS`; no FMA, reassociation or approximate arithmetic is introduced. Checked dimensions and disjoint destination/input footprints guard the assembly entrypoint. Other architectures, disabled AVX2 and non-eight-aligned widths use a scalar fallback.
2. Cached attention forms each query's `q+biasU` and `q+biasV` float32 vectors once per head/query row, outside the source-row loop. Those vectors are stack-owned and do not change accumulation order, cache state or returned output ownership.
3. Cached depthwise convolution consumes row-major GLU and the prior channel-major history directly. It no longer transposes GLU, constructs padded input, allocates a separate depth output, or transposes that output back. It writes into the already owned depth destination and allocates only replacement history. Per-element nine-tap summation order is unchanged, and the prepared history is committed only after the full block/tower succeeds. Public `Encoder0ConvCache.Update` and its owned padded/depth outputs remain unchanged.

## Baseline and profiling

Baseline source: `a79812169688a8f150b7fd67d1006a9855fdfc75`, the deployed four-worker dense and immutable K/V-view implementation. Go 1.26.2, i5-1340P, four Go threads; pinned Nemotron ASR model and tokenizer from [the replacement record](nemotron-asr-sigma-trial-20260930.md).

The refreshed CPU profile puts blocked SGEMM at 72.0% flat sampled CPU. Cached attention is 9.1% cumulative, including cold relative projection, scalar score and value loops. CPU/allocation profiles include benchmark setup/loading and an extra benchmark calibration request: loading accounts for 4.75 GB of the 7.52 GB allocation profile. They are attribution tools, not inference-only cost fractions or latency samples.

The updated candidate targets repeated attention work and convolution scratch traffic without modifying the already optimised GEMM path or retry semantics.

## Matched measurements

Native trials used four CPU quota, 8 GiB memory cap, no container swap, network disabled and no GPU device mounts. A one-second guard checked live Qwen slot activity and at least 6 GiB host available memory, stopping only the experiment on activity/unobservable slots or low memory. Resource ownership was handed off briefly for Qwen's authorised runtime update, then explicitly released back before these comparisons.

Incremental JFK screening, two requests per sample, two samples per arm:

| Comparison | Baseline mean | Candidate mean | Interpretation |
| --- | ---: | ---: | --- |
| Original → exact AVX2 value sum | 4.4435 s | 4.2856 s | 3.6% shorter in these samples |
| Value sum → value sum + direct convolution cache | 4.3465 s | 4.2919 s | Small timing change; about 55 MB less allocation |
| Value/conv → value/conv + hoisted query bias | 4.2188 s | 4.1294 s | 2.1% shorter in these samples |

These batches had different temperature/scheduling conditions. Do not add their percentages or compare the slowest early arm to the fastest later arm.

Final comparison repeated the deployed baseline against the combined candidate in interleaved order, three complete JFK requests per sample:

| Arm | Sample 1 | Sample 2 | Mean |
| --- | ---: | ---: | ---: |
| Deployed baseline | 4.1801 s | 4.1748 s | 4.1775 s |
| Combined candidate | 3.9515 s | 3.9807 s | 3.9661 s |

Measured time reduction: **5.06%**. Cumulative allocation changed from 963,344,539 to 908,292,933 bytes/request: **55,051,606 bytes (5.71%) less**, with about 1,680 fewer allocations. These are allocation volume, not peak RSS/model-size changes. Request benchmarks exclude model/WAV loading. Short samples do not establish a sustained/corpus-wide gain.

Separate fresh-stream recordings:

| Clip | Baseline | Candidate | Decisions |
| --- | ---: | ---: | ---: |
| JFK, 11 s | 4.543 s | 4.354 s | 185 |
| Podcast crop, 20 s | 7.938 s | 7.228 s | 419 |
| Fresh JFK repeat | 3.976 s | 3.789 s | 185 |

Every decision ID, absolute encoder frame and decoded text was compared and unchanged. This is implementation parity against the previously deployed native path, not independent PyTorch or labelled quality qualification.

## Verification

- AVX2/scalar value accumulation float32-bit exactness: 1/3/4/56/60 source rows, widths 7/8/128/135; input/destination-tail preservation, invalid dimensions, integer-overflow and alias rejection.
- Direct convolution matches public cache outputs bit-for-bit across frame lengths 1/3/4/5, identical history, prepared-copy rollback and nonfinite rejection.
- Full `go test -p 1 ./...`, `go vet -p 1 ./...`, `go build -p 1 ./...` passed with NVIDIA disabled and CGO disabled.
- `GODEBUG=cpu.all=off` focused tests passed.
- Linux ARM64 SIMD and ASR test binaries cross-built; this is compile-only fallback evidence, not native ARM timing.
- Released CPU shared-weight independent streams and cancellation-after-chunk tests passed; cancellation returns no partial decisions.
- `git diff --check` passed; no tolerance was relaxed. No C compiler is installed, so this pass has no race-detector evidence.

## Release state

The accepted candidate is ready for commit and idle-queue service verification. Native timings do not establish a live service speedup until deployment is tested. Local profiles, trial outputs, interleaved logs and guard records are under `tmp/nemotron-asr-opt3-20260930/`. Model assets and audio remain excluded from Git. Live Qwen's separately authorised update was outside this change; no Qwen settings or GPU recovery were performed by this pass.
