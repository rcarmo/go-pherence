# Matmul benchmark results

Generated baseline and candidate runs live below this directory. Each run contains `metadata.json`, append-only `records.jsonl`, and raw Go benchmark output.

The committed `baseline-361622d1` snapshot is a short validation capture (`200ms`, count 2) proving the runner and dense shape matrix on the i7-12700 host. Authoritative comparisons use the protocol defaults (`500ms`, count 5) and record end-to-end fixtures separately.

Run:

```bash
bun scripts/matmul-baseline.ts --suite dense,quant-cpu,gguf \
  --procs 1,2,all --out benchmarks/matmul/<name>
```

See `docs/matmul-benchmark-protocol.md` for acceptance rules.
