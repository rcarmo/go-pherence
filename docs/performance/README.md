# Performance

The [repository allocation/SIMD pass](allocation-simd-audit-20260919.md) combines an all-source lexical inventory with CPU profiles for sampling, tensor fusion, FFT/mel and tiny scoring. It records measured allocation reductions, noisy timing results, missing benchmarks and proposed work separately.

[Performance](performance.md) separates retained measurements from historical snapshots. Compare identical checkpoints, prompts, thread counts and execution phases, after checking numerical parity. Microbenchmarks and compile-only checks do not establish end-to-end performance.

* [Narrow CUDA Graph batch-1 benchmark](cuda-graph-benchmark.md)
* [Gemma4 E4B CPU SIMD gap](gemma4-cpu-simd-gap.md)
* [Matmul optimisation audit](matmul-audit.md)
* [Matmul benchmark protocol](matmul-benchmark-protocol.md)
* [Matmul optimisation results](matmul-optimisation-results.md)
* [Performance](performance.md)
* [SIMD inference matmul policy](simd-matmul.md)
* [TurboFieldfare adoption results](turbo-fieldfare-adoption-results.md)
* [TurboFieldfare audit](turbo-fieldfare-audit.md)
* [Whisper on RISC-V (SpaceMIT K1/K3) — RVV + IME optimization](whisper-riscv-optimization.md)

[Documentation index](../README.md)
