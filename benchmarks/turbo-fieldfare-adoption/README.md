# TurboFieldfare adoption benchmark scaffold

This directory holds the baseline measurement scaffold for the first TurboFieldfare-inspired MoE experiments in `go-pherence`.

Scope in this slice:

- deterministic `ExpertPool` route replay across global pool keys and slot sizes
- synthetic selected-expert compute benchmarks for generic MoE vs. per-expert chains
- no LFU policy work yet
- no new CUDA kernels yet

## Commands

```bash
go test ./backends/nvidia/runtime -run '^$' -bench 'BenchmarkExpertPoolReplaySyntheticGlobalKeys' -benchmem

go test ./model -run '^$' -bench 'BenchmarkMoESelectedExpertComputeSynthetic512x1024Top4' -benchmem
```

When CUDA SGEMM is available, the selected-expert benchmark compares the warm-pool generic GPU MoE path with its direct per-expert GPU chain. Otherwise it falls back to a synthetic CPU comparison.

See `metadata.json` for the captured host/GPU summary and current asset availability.
