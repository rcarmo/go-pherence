# Matmul benchmark protocol

This protocol is the acceptance gate for the optimisation programme in [matmul-audit.md](matmul-audit.md). A kernel change is retained only when it passes its numerical/model fixtures and improves the relevant distribution of shapes. One favourable microbenchmark is not sufficient.

## Environment capture

Every run records:

- git commit and dirty state;
- timestamp, hostname, OS, kernel, architecture, Go version, and build tags;
- CPU model, online CPU count, cache topology, NUMA topology, and available ISA flags;
- `GOMAXPROCS`, CPU affinity, governor, and visible frequency controls;
- GPU/accelerator model, driver, memory, and runtime availability;
- benchmark command, benchtime/count, fixture/model paths, and model revisions.

The runner never changes the CPU governor automatically. Use `--governor performance` only when the host exposes cpufreq and the caller has permission; otherwise the report records `unavailable`. CPU runs use `taskset` when present. GPU runs are serialized with `/tmp/go-pherence-gpu.lock`.

## Repetition and warm-up

- Kernel benchmarks: one unrecorded warm-up followed by `-benchtime=500ms -count=5 -benchmem` by default.
- Expensive real-model benchmarks: one warm-up only when model load can be separated from inference, then at least three recorded runs. Otherwise record model-load and inference stages separately from command stderr.
- The runner preserves raw Go benchmark output and emits JSONL metadata/status records. Comparisons use medians and report min/max or dispersion; do not compare only the best run.
- Run the worker matrix at `GOMAXPROCS=1`, `2`, and all visible CPUs. Accelerator-specific runs use their documented worker count as an additional case.

## Shape matrix

Dense and quantised linear kernels cover:

- decode: `M=1`;
- small prefill/verifier: `M=2,4,8,16,32`;
- MOSS multimodal prefill: `M=227`;
- Whisper encoder class: `M=1500`;
- representative square and expansion/contraction dimensions: `K,N=1024`, `1024x3072`, `3072x1024`, `1536x2048`, and model-native dimensions;
- non-vector and non-tile tails for M, N, and K.

Each datatype family adds its natural block/group tails and packed-layout cases. Decode and prefill results are reported separately.

## Baseline suites

The machine-readable runner groups available tests into:

- `dense`: F32 NN/NT/GEMV and BF16 dot/linear paths;
- `quant-cpu`: GPTQ Q4, MLX Q4, NVFP4, FP8;
- `gguf`: Q4_0, Q5_0, Q8_0, Q4_K, Q6_K plus available Q2_K/Q3_K model paths;
- `nvidia`: F32, Q4, MLX, GGUF, FP8 and NVFP4 CUDA/PTX tests/benchmarks;
- `spacemit`: RVV, IME2, AICPU and board paths;
- `models`: representative model-level kernel and graph-cycle benchmarks;
- `end-to-end`: exact-output CLI/model fixtures.

Unsupported hardware or missing model assets produce explicit `skipped` JSONL records with reasons. They are not counted as successful measurements.

## Profiles and counters

For representative baseline and retained implementations:

- collect Go CPU profiles and allocation profiles;
- use `perf stat` for cycles, instructions, branches, branch misses, cache references/misses and page faults when permitted;
- record elapsed/user/system time and maximum RSS;
- record NVIDIA stage/kernel timings, transfer counters and memory usage exposed by existing diagnostics;
- record RVV/IME2/AICPU queue, packing and accelerator counters exposed by their runners.

A counter marked unavailable remains in the report with the permission/tool failure.

## Correctness and acceptance

Before accepting a result:

1. malformed-input, overflow and tail tests pass;
2. scalar/reference numerical parity passes at the existing tolerance;
3. architecture cross-builds pass;
4. affected real-model token/transcript/image/embedding fixtures pass;
5. no race, resource-lifetime or allocation regression invalidates the comparison;
6. median performance improves on the target shapes without a material regression on decode or other default shapes;
7. packing time and additional resident memory are reported, including break-even reuse count.

The cumulative results file records baseline commit, candidate commit, hardware, suite/case, median, range, speedup, memory delta, parity result, disposition (`retained`, `reworked`, or `rejected`), and notes.
