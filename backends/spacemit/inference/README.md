# backends/spacemit/inference

Mid-level numeric building blocks layered on the `ime2` kernels. A thin functional
API — the full transformer runtime lives in `aicpu`.

## Exposed ops

- `QuantizeF32ToINT8`, `PackActivation`, `PackActivationInto` — activation quantization
- `RMSNorm`
- `MatVecQ4K`, `MatVecQ4KParallel` — Q4_K mat-vec (serial / threaded)
- `MatVecINT8Parallel`, `MatVecINT8Pool` — int8 mat-vec (threaded / `ime2.WorkerPool`-backed)

Depends on `ime2`; does not depend on `aicpu`.
