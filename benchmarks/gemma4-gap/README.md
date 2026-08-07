# Gemma4 CPU performance gap

This programme compares `go-pherence` with a CUDA-disabled rcarmo/llama.cpp b607 (`065d9d501`) build on the same Gemma4 E4B QAT Q4_0 GGUF. The comparison is intentionally awkward: timings must describe the same phases, and Go's activations, logits, generated tokens, K/V state and FP32 lane reduction order must not move by a bit merely to make a chart look better.

The earlier 447.17 prompt tok/s result did exactly that last part wrong in a more mundane fashion--it came from a CUDA-enabled binary where `-ngl 0` did not produce a CPU-only execution. It is retained as [`llama-cuda-ngl0-invalid.json`](llama-cuda-ngl0-invalid.json), with an unambiguous name, and is not an oracle.

## Frozen request

The model is `models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf`, running on six pinned Intel Core i7-12700 CPUs (`taskset -c 0-5`, `GOMAXPROCS=6`). The request has these exact boundaries:

* Prompt preparation supplies 123 copies of token `10979`; Gemma4 adds BOS token `2`, giving 124 evaluated prompt tokens. `BeginPrefill` allocates and prepares the request outside the timed section, while `PrefillNext(124)` is the prompt measurement.
* Prefill returns a boundary token before generation timing begins. The request exposes 48 output tokens, but only 47 subsequent model evaluations are timed.
* The frozen Go trajectory is `[70990 4941 159371 1390 67109 1485 237064 9120 70846 9120 13063 1390 236743 40594 1390 238007 1390 238122 4829 1390 1390 237307 1390 113858 9222 1390 9222 246293 9222 1390 1390 9222 783 82106 9222 246293 236929 4862 237275 1390 237083 237249 244359 238280 236929 395 236782 236929]`. The real-model gate checks all 48 IDs.

llama.cpp does not select that trajectory from the repeated-token prompt--its first greedy token is `10979`, whereas Go's is `70990`. For a useful performance comparison, the temporary b607 audit driver replays the 47 Go-selected pending tokens after evaluating the same prompt. It scans and synchronises logits on every step but ignores llama.cpp's selected ID. This measures identical matrix shapes and token-dependent MoE routing without pretending that the two implementations have cross-runtime logit parity.

## The CPU-only oracle

The audit build has no GPU backend (`"backends": "CPU"`, `"gpu_info": ""`, `n_gpu_layers: 0`). It uses a context sized for the combined 124-prompt/47-evaluation request, CPU flash-attention auto-selection, and F32 K/V storage. b607's command-line parser does not accept `f32` for `-ctk` or `-ctv`, so the audit source sets `GGML_TYPE_F32` as the local default before rebuilding.

`llama-bench` normally adds its independent prompt and generation cases when `-pg` is supplied. Passing `-p 0 -n 0` leaves only the combined phase:

```bash
taskset -c 0-5 /workspace/tmp/llama-b607-cpu/build-cpu/bin/llama-bench \
  -m "$MODEL" -p 0 -n 0 -pg 124,47 -r 3 -t 6 -ngl 0 -fa auto -o json
```

Six F32-KV samples from two adjacent runs are pooled because this host has conspicuous shared-load outliers. The median is the mean of samples three and four after sorting:

| Phase | Sample times (s, sorted) | Accepted median | 98% gate |
|---|---|---:|---:|
| 124-token prompt | 1.2766, 1.3168, 1.3277, 1.3907, 1.4274, 1.4283 | 91.2296 tok/s | 89.4050 tok/s |
| 47 generation evaluations | 4.3892, 4.3987, 4.4116, 4.5183, 6.7104, 7.5740 | 10.5265 tok/s | 10.3159 tok/s |

The temporary source changes are captured in [`audit/llama-b607-exact-cpu.patch`](audit/llama-b607-exact-cpu.patch). Raw phase telemetry is in [`audit/cpu_exact_go_trajectory_f32kv_124x47_r3.log`](audit/cpu_exact_go_trajectory_f32kv_124x47_r3.log) and [`audit/cpu_exact_go_trajectory_f32kv_paired_r3.log`](audit/cpu_exact_go_trajectory_f32kv_paired_r3.log); the adjacent JSON files record the b607 revision, CPU backend, K/V types and context parameters. The 6.71 s and 7.57 s generation samples are why a single pretty run is not accepted here.

## Where Go now lands

The retained long-prefill path quantises Q8_0 activations across six contiguous worker spans, then feeds the established one-row/eight-token SoA AVX-VNNI kernel. Block values remain byte-identical to serial `quantizeQ8_0To`; the focused exact test compares every scale and quantised byte, and running that worker-spawning test under the race detector covers concurrent writes into disjoint spans.

Alternating baseline/candidate runs moved the complete prompt median from 30.932 to 38.691 tok/s (+25.1%). A subsequent three-run contiguous-span check produced 32.227, 38.808 and 39.741 prompt tok/s, with 38.808 tok/s as the median. Generation code is unchanged and measured 8.144, 8.753 and 9.344 eval tok/s in the same run.

| Runtime | Prompt | Generation | Oracle efficiency |
|---|---:|---:|---:|
| CPU-only llama.cpp b607, exact phases | 91.230 tok/s | 10.526 eval tok/s | 100% / 100% |
| Go, retained median | 38.808 tok/s | 8.753 eval tok/s | 42.5% / 83.2% |

The exact workload is therefore not within the 2% target. Decode has a comparatively small 16.8% gap, while prefill is short by 57.5%.

## Why the prompt gap is stubborn

The useful techniques are already in the retained path: native AVX2/VNNI byte dots, row grouping for decode, one-row/eight-token activation tiling for prefill, SoA Q8 activation layout, six-way static row scheduling, vectorised Q6 coefficient expansion and now parallel contiguous activation quantisation. Four-row/two-token, flat-interleaved activation and persistent-worker variants were measured and removed because they regressed the complete request even when an isolated kernel looked promising.

The latest profile attributes 83.48% of flat CPU samples to `dotQ4_0Q8_0Tokens8SoAVNNI`. Serial activation work was worth removing, but Amdahl's law has become rather blunt: eliminating every remaining non-Q4 sample would only move 38.8 to roughly 46.5 tok/s. Closing the rest requires a materially faster exact Q4 matrix tile.

llama.cpp's prompt kernels can tile rows and tokens while choosing their own integer/FP32 accumulation scheme. Go deliberately retains eight independent FP32 lane accumulators and the legacy reduction order for every output. The tested four-row/two-token tile could not amortise both weight and activation traffic enough to compensate for that register pressure; the one-row/eight-token tile remains the fastest exact arrangement on this AVX2/VNNI machine. Wider reordering is not promoted unless all activations, logits, tokens and K/V state remain exact.

## Reproducing the Go side

```bash
MODEL="$PWD/models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
taskset -c 0-5 env \
  GOMAXPROCS=6 GOTMPDIR="$PWD/.gotmp" \
  GO_PHERENCE_GEMMA4_GAP_REAL=1 \
  GO_PHERENCE_GEMMA4_MAIN="$MODEL" \
  go test ./model -run '^TestGemma4RealCPUGap124x48$' \
  -count=3 -v -timeout=10m
```

Promotion also requires the focused exact Q4/Q8 tests, `go vet`, serialised race tests, the full package suite, and arm64/riscv64 compile checks. CUDA work stays out of this programme until the CPU path reaches the 2% gate--otherwise GPU numbers merely make the unresolved CPU kernel harder to see.
