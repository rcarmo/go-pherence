# Needle engram-width parity -- 2026-09-20

The source-checkpoint width slice now has an engram-bearing upstream fixture: width 1,024 to 512, two n-gram orders, four to two heads per order, and the parent's seed stride retained at four. Every selected child tensor matches the pinned upstream slice exactly. FP32 and CQ logits, all three auxiliary heads, and cached/full-prefix outputs pass the existing tolerances on Intel and native ARM.

This closes the bounded engram-width numerical gap. It does not establish released-model width quality, calibration, arbitrary-width support or packed-only execution. The [head-training/width guide](../guides/needle-head-training.md) still defines which source checkpoints can be sliced.

## The rounding difference

The previous width fixture had no engrams. The new synthetic fixture exposed two CQ logit mismatches at the existing `4e-5` absolute / `2e-3` relative tolerance; FP32 and exact tensor selection already passed. Dequantized weights differed by at most approximately `4e-8`, but replacing only the token embedding with Go's dequantized tensor reproduced the logit mismatch. Using upstream's dequantized weights with Go's A8/KV8 inference passed.

Upstream CQ uses multiplication by a dense, already-normalized Hadamard matrix. Go used a fast Walsh butterfly followed by normalization. These are equivalent over real numbers, but the sums and products round differently in FP32. Here those differences crossed a later activation-quantization threshold. A straight-through-estimator subtraction/addition diagnostic did not resolve the mismatch; matching the transform's pre-scaled products did.

Source CQ reference preparation now calls the reusable SIMD-dispatched `Walsh128ReferenceTo` in `backends/simd/runtime`. It uses a fixed 128-by-128 matrix and checks source/destination bounds and overlap before writing. Matrix storage is 64 KiB, shared read-only after initialization. The existing packed-archive butterfly, archive materialization and packed inference are unchanged. There is no bit-exact cross-ISA promise; the unchanged numerical comparisons pass on both tested machines.

## Reproducible fixture

[`scripts/needle-engram-width-reference.py`](../../scripts/needle-engram-width-reference.py) loads Needle 3 source at `fc5bae0f9b6138828fe7589f6b531fb9a26968de` on JAX CPU only. It emits parent, child and CQ-dequantized child tensors plus logits and head outputs. Natural engram geometry and stable Hadamard block size require the 1,024-to-512 rung for this two-order fixture; 512-to-256 fails upstream's block-geometry assertion.

The fixture has two layers, one engram layer, 17 slots, four attention heads in the parent, fixed QK/V dimensions of four and two mHC lanes. Deterministic nonzero, head-distinguishing matrices keep the complete tensor evidence compressible: 60,745,194 bytes of JSON occupy 3,104,101 gzip bytes. These are synthetic test weights, not a trained model or performance representative. The Go fixture reader caps decompression at 64 MiB.

```sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/cpu-jax-venv/bin/python scripts/needle-engram-width-reference.py \
  --upstream /path/to/pinned/needle \
  --output model/needle/testdata/needle3-engram-width.json.gz

GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/needle \
  -run '^TestWidthEngramUpstream$' -count=1 -timeout=120s -v
```

The test checks exact child tensor selection, CQ tensor parity, original parent ownership, preserved hash-seed geometry, FP32/CQ logits and head outputs, and each cached decoding prefix. Existing width tolerances are unchanged (`4e-5` / `2e-3` for logits, `1e-5` / `3e-3` for heads). A second inference check uses the upstream dequantized tensors to keep preparation and inference diagnosable separately.

## Cost of the reference transform

These one-group microbenchmarks exclude model loading, whole-tensor preparation, inference and training. The reference has O(128²) work rather than the butterfly's O(128 log 128), even though matrix SIMD makes the measured gap smaller on these hosts.

| Host | Packed butterfly | Dense reference | Allocation |
|---|---:|---:|---:|
| Intel i7-12700 | 389.3 ns | 544.9 ns | 0 bytes / 0 allocations |
| Native ARM64 CIX P1 | 749.5 ns | 1,525 ns | 0 bytes / 0 allocations |

```sh
go test ./backends/simd/runtime -run '^$' -bench '^BenchmarkWalsh128$' \
  -benchmem -benchtime=500ms
```

No released-model speed claim follows from this. Source CQ preparation pays the reference cost; deployed archive inference retains its current implementation.

## Validation and preservation

Affected-package race tests and CPU-feature-disabled runs (`GODEBUG=cpu.all=off`) passed. Native ARM tests used `GOMAXPROCS=2 nice -n 10` and passed **105 Needle model, 48 loader, eight CLI and 565 SIMD tests/subtests**. Those are actual native executions. RISC-V validation is cross-compilation only.

Final `GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s` completed with exit zero: **114 packages passed, 54 without tests**. Two earlier combined/outer command timeouts did not supply a trustworthy exit result; the final invocation used a longer shell deadline without changing package test timeouts. `go vet ./...`, `go build ./...`, the Linux/RISC-V cross-build, and `make docs-check` passed (**363 Markdown files, zero broken links**). All **16 freeze entries**, both preserved binaries and **676 evaluation record hashes** are unchanged. No numerical tolerance was increased, no GPU was queried or used, and no service was changed. Independent delegation has remained unavailable; a read-only DiffusionGemma assessment timed out without a usable report and supplies no review coverage for this change.
