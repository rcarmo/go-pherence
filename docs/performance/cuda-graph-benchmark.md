# Narrow CUDA Graph batch-1 benchmark

`backends/nvidia/runtime` now contains a live CUDA Graph parity test and benchmark for a deliberately small fixed-shape batch-1 segment built from existing capture-safe ops:

- `DevRMSNorm`
- `DevScale`
- `DevAdd`

The benchmark compares:

- eager warm launches
- captured warm graph launches
- capture+instantiate cost

Run the live parity test only when a CUDA device is available and you explicitly opt in:

```bash
GOTMPDIR=$PWD/.gotmp GO_PHERENCE_RUN_CUDA_GRAPH_LIVE=1 go test ./backends/nvidia/runtime -run TestCUDAGraphBatch1SegmentLiveParity -count=1
```

Run the focused benchmark:

```bash
GOTMPDIR=$PWD/.gotmp go test ./backends/nvidia/runtime -run '^$' -bench BenchmarkCUDAGraphBatch1Segment -benchmem
```

This is **not** a claim of full Gemma4 token capture. The narrow benchmark exists to validate basic graph replay with stable device-resident buffers while the larger runtime still has graph blockers, including:

- host `Data()` / CPU-side tensor access in hot paths
- per-step allocations or lazy uploads during execution
- CPU fallbacks and KV/cache shadow copies that break all-GPU capture
- logits download / host-side consumption at token boundaries

Treat it as a low-level capture probe, not end-to-end model graph coverage.

## RTX 3060 result

Measured on 2026-08-07 with an NVIDIA GeForce RTX 3060, `GOMAXPROCS=2`, a 1024-element fixed shape, three 2-second benchmark repetitions, and synchronization included in each iteration:

| Mode | Warm latency | Allocations |
|---|---:|---:|
| Eager three-kernel segment | 12.50–13.42 µs | 1,344 B / 55 allocs |
| Captured graph replay | 9.94–10.01 µs | 232 B / 14 allocs |
| Capture + instantiate + destroy | 13.14–14.05 µs | 1,888 B / 84 allocs |

Replay produced exact output parity (`maxDiff=0`) and reduced this micro-segment's synchronized warm latency by roughly 20%. The shape-key guard rejects mismatched replay and teardown is idempotent.

A dedicated non-blocking capture stream is required by the current CUDA driver/runtime combination: attempting capture on the legacy default stream failed with CUDA error 900 (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). Kernel routing changes only while this explicitly serialized experimental capture API is active; production inference does not enable graph capture.

Disposition: retain the low-level probe and primitives, but do not integrate graph replay into Gemma4 generation until the host interactions above are removed and a complete fixed-shape token segment can be measured against eager execution.
