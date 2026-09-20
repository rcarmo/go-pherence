## Needle 3 head gradients and half-width validation

The new head-training path updates one auxiliary head with frozen trunk cells. Its loss and gradients match the pinned JAX architecture for three explicit objectives: confidence binary cross-entropy, router cross-entropy, and MSE on normalized embeddings. These are supervised primitives, not a claim to reproduce Cactus's training dataset, batch recipe or calibration.

The [usage guide][guide] specifies accepted targets, source-checkpoint requirements and CLI commands. The source reference remains Needle 3 commit `fc5bae0f9b6138828fe7589f6b531fb9a26968de`, already pinned by the existing model fixtures.

## Numerical checks

`scripts/needle-head-width-reference.py` imports the pinned architecture locally and runs on CPU. For each head it compares FP32 and CQ-W4/A8/KV8 loss and every nonzero JAX parameter gradient. Missing trunk gradients must be zero, because upstream stops gradients at `_head_cells`. Tests also assert that training updates no other head or trunk tensor, reduces each supplied objective, and reproduces the output after safetensors save/load.

The half-width reference is a width-16, two-layer source model trained with a width-8 rung. It has split permutations, four query heads, one KV head, nontrivial taps and all three auxiliary heads. The Go slice matches every upstream selected tensor exactly; child logits and head outputs match in FP32 and CQ. Parent tensors remain unchanged.

This tiny width reference has no engrams. The implementation follows upstream's engram cuts but rejects incompatible fixed-head configurations; an auto-head, engram-bearing width fixture remains an additional qualification task. Width slicing is restricted to the requested half-width rung being present in `ladder_widths`, compatible Hadamard factors, and source weights. Arbitrary narrowing, archive slicing and AB-scale width remapping are not admitted.

Invalid target values/shapes, zero-norm embedding targets, all-padding input, missing heads, workspace exhaustion, unsupported source types and unlisted/invalid widths are tested. The CLI test exercises width selection, head update, new checkpoint creation and subsequent head inference. Confidence/router results remain raw and uncalibrated.

## Platforms and allocations

The final whole-tree NVIDIA-disabled race run passed for 114 packages (54 had no tests). Host vet/build, RISC-V cross-builds and AVX2/FMA-disabled Needle tests passed; documentation checks scanned 360 Markdown files without broken links. Native ARM64 CIX P1 test executables passed 81 model tests/subtests and seven CLI tests. Native runs use `GOMAXPROCS=2`, `nice -n 10` and bounded timeouts on the existing shared board; no services or GPU state were changed. Final verbose logs are retained under `/workspace/tmp/needle-port/`.

A short Intel i7-12700 benchmark of confidence-head gradients over the tiny padded five-token fixture measured about **0.185 ms, 238,074 bytes and 2,666 allocations** per call. This includes recomputing frozen cells, not just the small projection. There is no production-model training throughput or peak-memory claim. Full-model tape, optimizer state and source checkpoint residency remain separate from the logical workspace cap.

The code reuses existing SIMD matrix, vector and gradient operations. Frozen trunk execution does not construct a backward tape; the head tape does. An independent review delegate timed out and returned no findings, so it is not counted as additional coverage.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 python scripts/needle-head-width-reference.py \
  --upstream /path/to/pinned/needle \
  --output model/needle/testdata/needle3-head-width.json
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./model/needle ./cmd/needle
go test ./model/needle -run '^$' -bench '^BenchmarkNeedleHeadGradient$' -benchmem
```

Use the same JAX 0.11.2, Flax 0.12.9 and NumPy 2.5.3 fixture environment documented in the [model README][model]. Neither fixture generation nor Go execution downloads a model. A trained confidence score needs separate held-out calibration before it can be treated as a probability; this work does not provide that calibration.

[guide]: ../guides/needle-head-training.md
[model]: ../../model/needle/README.md
