# Gemma4 performance-gap programme

This programme uses llama.cpp as the implementation oracle for the same local Gemma4 E4B QAT Q4_0 GGUF. Optimisations are promoted only when output parity holds and the complete prompt/decode workload improves; isolated kernel gains are supporting evidence rather than acceptance.

## Frozen workload

* Model: `models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf`.
* Hardware: Intel Core i7-12700 allocation with 6 cores/threads and an RTX 3060 12GiB.
* Shape: 124 prompt tokens and 48 greedy output tokens, matching the compact agentic tool-selection workload.
* rcarmo/llama.cpp build: b607 (`065d9d501`), from the `rcarmo-spacemit/main` worktree.
* Repetitions: three; CPU uses six threads and no offload, GPU uses all available layers.

```bash
llama-bench -m "$MODEL" -p 124 -n 48 -r 3 -t 6 -ngl 0 -o json
llama-bench -m "$MODEL" -p 124 -n 48 -r 3 -t 6 -ngl 99 -o json
```

## Oracle baseline

| Runtime | Backend | Prefill | Decode |
|---|---|---:|---:|
| rcarmo/llama.cpp | CPU | 447.17 tok/s | 9.31 tok/s |
| provisional system llama.cpp | RTX 3060 | 2,182.85 tok/s | 83.54 tok/s |

Raw results are in [llama-cpu.json](llama-cpu.json) and [llama-gpu.json](llama-gpu.json). The GPU result remains provisional and is not an acceptance oracle. The Intel gates are at least 438.22 tok/s prefill and 9.12 tok/s decode -- 98% of the fork measurements.

The earlier go-pherence MLX4 run is not a cross-runtime comparison because llama.cpp cannot load that format directly. It remains useful as a go-pherence backend check: the RTX 3060 produced 19.3 tok/s while CPU produced 1.4 tok/s, with byte-identical tool-call output.

## First profile

A real-GGUF request-scoped SIMD session completed its frozen one-token gate in 8.19s. CPU samples identify two scalar quantised row-dot loops as the immediate decode gap:

| Function | Flat CPU samples |
|---|---:|
| `loader/gguf.DotQ4_0Q8_0` | 68.80% |
| `loader/gguf.DotQ6KQ8K` | 15.43% |
| Linux syscalls | 8.57% |

`gemvRowsParallel` accounts for 87.31% cumulatively. The Q4_0 function currently reproduces llama.cpp's AVX2 reduction order in scalar Go; it does not call an AVX2 kernel. The first implementation slice is therefore a guarded packed Q4_0×Q8_0 amd64 primitive with exact-reference and portable-fallback gates, followed by Q6_K×Q8_K.

## Promotion gates

* Exact frozen generated-token parity and focused scalar-oracle parity.
* No regression on unsupported architectures; arm64 and riscv64 cross-builds remain green.
* Improvement in complete prefill/decode timing, not only the row-dot microbenchmark.
* CPU and GPU results remain separate. NVIDIA work must retain quantised GGUF weights on device and eliminate measured host synchronisation before it can be compared with llama.cpp CUDA.
* Stateful NVIDIA request sessions remain blocked until each request owns its device KV/cache state; legacy single-request timing must not be presented as serving parity.

## Packed Q4_0 x Q8_0 AVX2 slice

The first amd64 kernel preserves the existing eight-lane FMA and reduction order and retains the scalar implementation for non-amd64 builds. On the i7-12700, the 2560-element row-dot improved from 2.24-3.17us to 0.22-0.44us across three focused samples. The frozen real-GGUF two-token session gate improved from 8.19s to 4.22s and retained exact output. A follow-up profile moves the primary flat hotspot to scalar `DotQ6KQ8K` at 45.13%; packed Q4_0 remains 20.29% and syscall/model-loading samples account for 22.56%. This is retained as an intermediate slice but does not satisfy the 98% end-to-end gate.
