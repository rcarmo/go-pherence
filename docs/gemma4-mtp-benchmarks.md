# Gemma4 QAT + MTP validation and benchmark snapshot

Snapshot date: 2026-06-16

This page records the current Gemma4 QAT + MTP correctness/performance benchmark state. The strict llama.cpp selected-logit gate remains the known blocker for declaring 1:1 graph fidelity.

## Validation matrix

| Dimension | Gate | Status | Evidence |
|---|---|---:|---|
| Output / accepted-token parity | `make gemma4-mtp-parity GOTMPDIR=$PWD/.gotmp` | PASS | `TestGemma4MTPLlamaCPPParityFixture` passes on the default accepted-token fixture; standalone runner reports `matched true`. |
| Strict llama.cpp selected-logit parity | `GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=tmp/gemma4-mtp-llamacpp-fixture.json make gemma4-mtp-strict-parity GOTMPDIR=$PWD/.gotmp` | FAIL | Six stable verifier selected-logit mismatches remain against the local `llama.cpp --flash-attn on` fixture. Keep `RealAssetAcceptanceParity=false`. |
| SIMD / GGUF quant scalar oracles | `go test ./loader/gguf -run 'TestDequantRowQ4KToZeroBlock|TestDequantRowQ4KToMatchesGGMLNibbleGroups|TestExpertMatricesQ4KGemvMatchesDequantScalar|TestDequantRowQ8_0ToMatchesScaleTimesInt8|TestQuantizeQ8_0UsesRoundAwayFromZeroWithUnroundedScale|TestDotQ4_0Q8_0MatchesScalarReference|TestQuantizeQ8KComputesScaleQuantsAndBlockSums|TestDequantRowQ6KToMatchesScalarReference|TestDotQ6KQ8KMatchesScalarReference' -count=1 -v` | PASS | Verbose run shows real PASS lines for Q4_K, Q4_K expert GEMV, Q8_0, Q4_0×Q8_0, Q8_K, Q6_K, and Q6_K×Q8_K coverage; no skips. |
| GPU ↔ CPU oracle for Gemma4/MTP verifier boundary | `flock /tmp/go-pherence-gpu.lock -c 'GO_PHERENCE_GPU_DEBUG=1 GOTMPDIR=$PWD/.gotmp go test ./model -run TestGemma4MTPVerifierPostAttentionRMSNormGPUParity -count=1 -v'` | PASS | Runs on RTX 3060, loads 78 PTX kernels, and validates CUDA `DevRMSNormOK` against CPU SIMD RMSNorm for the `attn_wo -> attn_post_norm` verifier boundary. The test fails rather than silently skipping when `nvidia-smi` sees a GPU but the NVIDIA runtime cannot initialize. |
| Required tagged Gemma4 GPU ↔ CPU compute parity | `make gemma4-gpu-cpu-parity GOTMPDIR=$PWD/.gotmp` | PASS | Runs the tagged `diagnostic gemma4fixtures` Gemma4 GPU tests under `flock /tmp/go-pherence-gpu.lock`, bounded by `GO_PHERENCE_GPU_KV_MAX_SEQ=64`, as separate processes to avoid CUDA shutdown/reinit crashes. Covers `TestGemma4GPUGenerate`, `TestGemma4QuantizedCPUvsGPULayerWalk`, `TestGemma4CPUvsGPUProjectionTrace`, `TestGemma4QuantizedCPUvsGPUOpTrace`, and `TestGemma4QuantizedCPUvsGPUOpTraceEarly`. |

Strict selected-logit mismatch set:

```text
row0 token236751 got=14.1096830368042 want=14.126071
row0 token236757 got=15.128488540649414 want=15.1981421
row1 token236751 got=7.351251125335693 want=7.34865713
row1 token236757 got=13.789809226989746 want=13.9382925
row2 token236751 got=20.40302085876465 want=20.2557411
row2 token236757 got=28.634824752807617 want=28.6220856
```

## Benchmark highlights

### Gemma4 MTP gate component timing

From `tmp/bench-gemma4-mtp-gate-components-20260616-193632.log`:

| Component | Result | Real | User | Sys |
|---|---:|---:|---:|---:|
| model fixture test | PASS | 28.259s | 23.254s | 3.813s |
| standalone runner | matched true | 30.562s | 24.116s | 3.973s |
| GGUF quant oracle subset | PASS | 0.257s | 0.175s | 0.065s |

The default parity gate is dominated by model load plus one graph decode cycle; quant oracles are negligible.

### CPU hot paths

From `tmp/bench-cpu-gemma4-core-count3-20260616-193535.log`:

| Benchmark | Mean | Min | Max | Allocations |
|---|---:|---:|---:|---:|
| `BenchmarkCPUHotRMSNorm3584` | 455.4 ns | 446.8 ns | 468.0 ns | 0 |
| `BenchmarkCPUHotGQAAttentionDecode512` | 226.141 µs | 207.894 µs | 235.426 µs | 0 |
| `BenchmarkCPUHotGemvQ4Sym1536x2048` | 2.623 ms | 2.613 ms | 2.635 ms | 0 |

Single-run hot-path timings vary materially with host load; use repeated samples for comparisons.

### GGUF qdot microbenchmarks

Harness: `loader/gguf/qdot_bench_test.go`.

From `tmp/bench-gguf-qdot-count5-20260616-193512.log`:

| Benchmark | Mean | Min | Max | Allocations |
|---|---:|---:|---:|---:|
| `BenchmarkDotQ4_0Q8_0` | 2652.8 ns | 2388 ns | 3059 ns | 0 |
| `BenchmarkDotQ6KQ8K` | 6281.0 ns | 6043 ns | 6710 ns | 0 |

### Experimental verifier batch-layer CPU path

`GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS=1` preserves accepted-token parity on the default fixture but is not public-ready:

| Fixture | Result | Real | Notes |
|---|---:|---:|---|
| default accepted-token runner | PASS (`matched true`) | 36.924s | Slower than default runner in that sample. |
| strict selected-logit fixture | FAIL | 44.270s | Different mismatch pattern; keep gated off. |

### GPU smoke and CLI timing

CUDA readiness smoke under the shared GPU lock:

```text
[gpu] NVIDIA GeForce RTX 3060 (28 SMs) — pure Go, no CGo
[gpu] All 78 kernels loaded in 1 module
[gpu] Streams + events initialized (prefetch overlap)
GPU rms_norm: OK
```

Gemma4 E4B MTP real-prompt GPU smoke (`tmp/bench-gemma4-e4b-mtp-gpu-smoke-20260616-194251.log`):

| Metric | Value |
|---|---:|
| main model load | 20.61s |
| prompt prefill | 0.72s for 10 tokens |
| drafter load | 0.906s |
| drafter step | 0.0336s |
| wall | 36.623s |

Fixed `llmgen` CPU-vs-GPU 4-token comparison (`tmp/bench-gemma4-e4b-fixed-cli-cpu-vs-gpu-gen4-20260616-215132.log`):

| Path | Output | Model load | Total time | Generation time | Tokens/sec | ms/token | Process wall |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU | `HelloHello! How can` | 43.95s | 8.52s | 6.81s | 0.6 | 1703.5 | 54.170s |
| GPU | `HelloHello! How can` | 5.14s | 0.73s | 0.58s | 6.8 | 146.1 | 11.150s |

The `llmgen` CLI output accounting fix is committed as `46edd1ea Fix llmgen generated token accounting`; CPU and GPU now report the same prompt/generated token counts for this smoke.

## GPU coordination

Use the shared GPU lock for new GPU validations/benchmarks:

```bash
flock /tmp/go-pherence-gpu.lock -c '<gpu command>'
```

Current agreed windows:

- Gemma4/MTP: minute `:20-:35`
- DiffusionGemma: minute `:00-:15`
- Whisper: minute `:40-:55`

Always verify `nvidia-smi` before starting a large model run and release the GPU promptly.

### MTP smoke with prompt-KV reuse

`tmp/bench-gemma4-e4b-mtp-smoke-kvreuse-20260616-220916.log` measures the supported real-prompt MTP smoke with the in-process prompt-context cache enabled:

```bash
flock /tmp/go-pherence-gpu.lock -c \
  "GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
    -gpu -gpu-layers 0 \
    -model models/gemma4-e4b-it-4bit \
    -mtp-drafter models/gemma4-e4b-mtp-drafter \
    -mtp-smoke -mtp-real-prompt \
    -mtp-kv-reuse -mtp-kv-repeat 3 \
    -prompt 'Hello'"
```

Result: PASS.

| Metric | Value |
|---|---:|
| main model load | 5.49s |
| drafter load | 0.218s |
| prompt prefill | 0.562s for 10 tokens |
| prompt KV cache hit | true |
| `kv_repeat` | 3 |
| drafter step | 0.0118s |
| process wall | 11.563s |

The smoke produced token `9259`, preserved `ready_for_experimental_generation=true`, and still reports public blockers `public_generation_wiring`, `real_asset_acceptance_parity`, and `full_layer_batch_verifier_default_enablement`.

### Experimental `-mtp-generate` benchmark status

Attempts to benchmark `llmgen -mtp-generate` with the local E4B MLX pair currently fail before generation because the prompt-context external-KV mapping supplies empty K/V for drafter layer 0:

```text
mtp generate: MTP drafter steps: MTP drafter step 0: drafter layer 0 external KV K/V=0/0, want 5120
mtp generate: MTP drafter steps: MTP drafter step 0: drafter layer 0 external KV K/V=0/0, want 5632
```

Logs:

- `tmp/bench-gemma4-e4b-mtp-generate-20260616-220751.log`
- `tmp/bench-gemma4-e4b-mtp-generate-2tok-20260616-220832.log`

So speculative-decode tokens/sec for the public CLI path is not yet benchmarkable on this asset pair. Use the llama.cpp fixture acceptance contract and `-mtp-smoke` drafter-step timings until this external-KV handoff gap is fixed.

### Gemma4 E4B GPU-only 8-token smoke

`tmp/bench-gemma4-e4b-gpu-gen8-20260616-221034.log` provides a bounded GPU-only generation timing after the `llmgen` accounting fix:

```bash
flock /tmp/go-pherence-gpu.lock -c \
  "GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
    -gpu -gpu-layers 0 \
    -model models/gemma4-e4b-it-4bit \
    -tokens 8 \
    -prompt 'Hello'"
```

Result: PASS.

```text
Output: HelloHello! How can I help you today
Prompt tokens:    1
Generated tokens: 8
Model load:       4.96s
Total time:       0.98s
Generation time:  0.87s
Tokens/sec:       9.2
ms/token:         108.5
Process wall:     11.204s
```

GPU memory returned to ~255 MiB after process exit.

### GGUF strict-fixture steady-state MTP graph-cycle throughput

`tmp/mtp_cycle_bench.go` was used as a temporary one-off benchmark to load the Gemma4 E4B QAT GGUF verifier and BF16 MTP drafter once, build the prompt context once, then run the strict fixture graph cycle five times from fresh decode states.

Log: `logs/mtp-cycle-steady-20260617-002600.log`

Result: PASS for accepted-token semantics on every iteration.

```text
iter=0 elapsed=11.893803613s output=[564 236789 236757] accepted=2 bonus=236757
iter=1 elapsed=11.826009510s output=[564 236789 236757] accepted=2 bonus=236757
iter=2 elapsed=11.845546187s output=[564 236789 236757] accepted=2 bonus=236757
iter=3 elapsed=11.919657652s output=[564 236789 236757] accepted=2 bonus=236757
iter=4 elapsed=11.851677704s output=[564 236789 236757] accepted=2 bonus=236757
```

Timing summary:

| Metric | Value |
|---|---:|
| model + drafter load | 3.574s |
| prompt context build | 5.249s |
| graph cycles | 5 |
| graph total | 59.337s |
| output tokens | 15 |
| steady-state graph throughput | 0.253 tok/s |
| mean graph cycle time | 11.867s |

This is the best current apples-to-apples MTP graph-cycle throughput number excluding model load. It is still CPU/GGUF verifier-bound and far from llama.cpp performance, but the accepted-token contract is stable.

### Steady-state MTP graph-cycle with gated verifier batch layers

`GO_PHERENCE_MTP_VERIFIER_BATCH_LAYERS=1` was measured with the same temporary steady-state graph-cycle harness.

Log: `logs/mtp-cycle-steady-batchlayers-20260617-002807.log`

Result: PASS for accepted-token semantics on every iteration, but this path is still not strict selected-logit safe.

```text
iter=0 elapsed=11.879827470s output=[564 236789 236757] accepted=2 bonus=236757
iter=1 elapsed=11.908227961s output=[564 236789 236757] accepted=2 bonus=236757
iter=2 elapsed=12.001955533s output=[564 236789 236757] accepted=2 bonus=236757
iter=3 elapsed=12.198878912s output=[564 236789 236757] accepted=2 bonus=236757
iter=4 elapsed=12.014943740s output=[564 236789 236757] accepted=2 bonus=236757
```

| Metric | Value |
|---|---:|
| graph cycles | 5 |
| graph total | 60.004s |
| output tokens | 15 |
| steady-state graph throughput | 0.250 tok/s |
| mean graph cycle time | 12.001s |

The gated batch-layer scaffold is slightly slower than the baseline steady-state run (`0.250 tok/s` vs `0.253 tok/s`) and remains less numerically faithful in strict selected-logit comparisons, so it should stay disabled by default.

### CPU profile of one strict-fixture MTP graph cycle

A one-cycle CPU profile was captured with a temporary `runtime/pprof` harness after loading the Gemma4 E4B QAT GGUF verifier and BF16 MTP drafter once.

Profile files:

- `logs/mtp-cycle-cpu.pprof`
- `logs/mtp-cycle-cpu-top-20260617-003003.txt`

Top samples:

```text
Duration: 11.83s, Total samples = 11.82s
flat      flat%   cum      cum%
7.55s     63.87%  7.82s    66.16%  loader/gguf.DotQ4_0Q8_0
3.88s     32.83%  3.90s    32.99%  loader/gguf.DotQ6KQ8K
0.11s      0.93%  0.11s     0.93%  encoding/binary.littleEndian.Uint16
0.10s      0.85%  0.19s     1.61%  half.F16ToF32
```

Call-path summary:

- `RunMTPVerifierBatchForward` → `forwardMTPPromptLayer` → Q4_0×Q8_0 GGUF projections account for roughly two thirds of the graph cycle.
- `FinishCPUDecodeBatch` → `LMHeadLogitsInto` → Q6_K×Q8_K tied LM-head dot accounts for roughly one third.

This confirms steady-state MTP graph throughput is dominated by quantized GGUF row-dot kernels, not MTP drafter overhead or acceptance accounting. Correctness work should stay on the baseline path, but performance parity with llama.cpp needs faster Q4_0×Q8_0 projection kernels and Q6_K×Q8_K LM-head kernels, likely batched/repacked to match llama.cpp's CPU backend strategy.

### Parallel GGUF row-GEMV performance update

`loader/gguf.GemvQ4_0Q8_0Rows` and `GemvQ6KQ8KRows` now parallelize large output-row loops across `GOMAXPROCS` workers. This preserves per-row dot numerics while improving the MTP verifier and LM-head throughput because the profile showed roughly 99% of steady-state graph-cycle samples in Q4_0×Q8_0 and Q6_K×Q8_K row-dot calls.

Validation:

```bash
make gemma4-mtp-parity GOTMPDIR=$PWD/.gotmp
make gemma4-gpu-cpu-parity GOTMPDIR=$PWD/.gotmp
```

Both passed after the change. Strict selected-logit values are unchanged from baseline.

Steady-state strict fixture graph-cycle throughput improved from `0.253 tok/s` to `1.144 tok/s`:

| Run | Mean cycle | Output tokens | Throughput |
|---|---:|---:|---:|
| before parallel row-GEMV | 11.867s | 15 / 5 cycles | 0.253 tok/s |
| after parallel row-GEMV | 2.622s | 15 / 5 cycles | 1.144 tok/s |

Strict selected-logit gate timing improved from about `0.141 tok/s` to `0.357 tok/s` while preserving the same remaining six selected-logit mismatches.

Log references:

- `logs/mtp-cycle-steady-20260617-002600.log` (before)
- `logs/mtp-cycle-steady-parallel-gemv-20260617-003209.log` (after)
- `logs/mtp-strict-parallel-gemv-20260617-003242.log`

### GOMAXPROCS scaling for parallel GGUF row-GEMV

`logs/mtp-cycle-gomaxprocs-scaling-20260617-004223.log` measures the strict-fixture MTP graph cycle after row-level parallelization while varying `GOMAXPROCS`.

| GOMAXPROCS | Mean cycle | Throughput |
|---:|---:|---:|
| 1 | 11.856s | 0.253 tok/s |
| 2 | 6.222s | 0.482 tok/s |
| 4 | 3.473s | 0.864 tok/s |
| 6 | 2.791s | 1.075 tok/s |
| 8 | 3.604s | 0.832 tok/s |

On the local i7-12700 container allocation, `GOMAXPROCS=6` is the best measured setting. Oversubscribing to 8 workers slows the MTP graph cycle despite more goroutines. Use this as the default comparison point for future tok/s measurements on this host.

### Strict selected-logit gate with best measured thread setting

After the parallel GGUF row-GEMV change, the strict selected-logit test was run with the best local thread setting from the scaling sweep:

```bash
GOMAXPROCS=6 GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=$PWD/tmp/gemma4-mtp-llamacpp-fixture.json \
GO_PHERENCE_GEMMA4_MTP_DRAFTER=$PWD/models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf \
GOTMPDIR=$PWD/.gotmp go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1 -v
```

Log: `logs/mtp-strict-parallel-gemv-gmp6-20260617-004505.log`

Result remains expected FAIL with the same six selected-logit mismatches, but the current strict-gate effective throughput is:

| Setting | Elapsed | Effective output tokens | Effective throughput | Sum abs error | Max abs error |
|---|---:|---:|---:|---:|---:|
| `GOMAXPROCS=6` | 9.897s | 3 | 0.303 tok/s | 0.397138 | 0.148483 |

This is slower than the isolated steady-state graph-cycle result (`1.075 tok/s` at `GOMAXPROCS=6`) because the strict `go test` path includes test harness/model setup overhead. The selected-logit values are unchanged from baseline, confirming the row-GEMV parallelization is performance-only.

### Best observed strict-gate throughput after parallel row-GEMV

A later strict selected-logit run with the committed parallel row-GEMV implementation and `GOMAXPROCS=6` produced the best observed strict-gate effective throughput so far.

Log: `logs/mtp-strict-current-gmp6-20260617-004858.log`

| Setting | Elapsed | Effective output tokens | Effective throughput | Sum abs error | Max abs error |
|---|---:|---:|---:|---:|---:|
| `GOMAXPROCS=6` | 7.679s | 3 | 0.391 tok/s | 0.397138 | 0.148483 |

The six selected-logit mismatches are unchanged, so the performance improvement remains numerically neutral relative to the baseline strict path.

### Additional strict-parity hypotheses ruled out after parallel row-GEMV

After the parallel row-GEMV performance change, several narrow correctness hypotheses were retested against the strict selected-logit fixture. All were worse than the committed baseline, so none should be adopted:

| Experiment | Effective tok/s | Sum abs error | Max abs error | Log |
|---|---:|---:|---:|---|
| Q4_0×Q8_0 no-FMA accumulation | 0.152 | 0.500195 | 0.149985 | `logs/mtp-strict-q4-no-fma-20260617-002234.log` |
| Q8_0 nearest-int rounding | 0.137 | 0.601656 | 0.216660 | `logs/mtp-strict-q8-nearest-20260617-002308.log` |
| Q8_0 rounded-scale inverse | 0.136 | 0.513401 | 0.221303 | `logs/mtp-strict-q8-rounded-id-20260617-002343.log` |
| LM-head-only dequantized projection | 0.358 | 0.412106 | 0.162596 | `logs/mtp-strict-lmhead-dequant-20260617-004120.log` |
| float64 final-logit softcap | 0.312 | 0.397139 | 0.148483 | `logs/mtp-strict-softcap-f64-20260617-005035.log` |
| no Gemma4 MTP verifier layer-output BF16 | 0.346 | 0.513223 | 0.304503 | `logs/mtp-strict-no-layer-bf16-after-parallel-20260617-005123.log` |

Current best strict baseline remains the committed parallel row-GEMV path with `GOMAXPROCS=6` (`0.391 tok/s`, sum abs error `0.397138`, max abs error `0.148483`).

### Refined GOMAXPROCS 5/6/7 sample

A later 5-cycle repeat around the best local thread setting showed the same ordering but lower absolute throughput under current host load:

Log: `logs/mtp-cycle-gomaxprocs-5-7-20260617-005955.log`

| GOMAXPROCS | Mean cycle | Throughput |
|---:|---:|---:|
| 5 | 3.551s | 0.845 tok/s |
| 6 | 3.157s | 0.950 tok/s |
| 7 | 3.876s | 0.774 tok/s |

`GOMAXPROCS=6` remains the best local setting among nearby values. The absolute tok/s varies materially with host load, so compare variants using adjacent runs and preserve the same `GOMAXPROCS` setting.

### Gemma4 layer scalar after BF16 in MTP verifier path

Moving Gemma4 MTP verifier layer scalar application after the final BF16 narrowing in `forwardMTPPromptLayer` materially improves strict selected-logit fidelity while preserving accepted-token parity.

Validation:

```bash
make gemma4-mtp-parity GOTMPDIR=$PWD/.gotmp
GOMAXPROCS=6 GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=$PWD/tmp/gemma4-mtp-llamacpp-fixture.json \
GO_PHERENCE_GEMMA4_MTP_DRAFTER=$PWD/models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf \
GOTMPDIR=$PWD/.gotmp go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1 -v
```

Log: `logs/mtp-strict-layer-scalar-after-bf16-final-20260617-010441.log`

| Path | Effective tok/s | Sum abs error | Max abs error |
|---|---:|---:|---:|
| previous best baseline | 0.391 | 0.397138 | 0.148483 |
| layer scalar after BF16 | 0.385 | 0.237515 | 0.093098 |

The strict selected-logit gate is still red, but this is the best correctness result so far and reduces total selected-logit error by about 40%.

### Layer-scalar ordering cutoff sweep

A cutoff sweep tested whether only early layers need the improved Gemma4 MTP verifier ordering (`BF16(hidden)` before `layer_output_scale`) while later layers retain the old ordering. Lower `sum_abs` is better.

| Cutoff (`layerIdx < N` uses scalar-after-BF16) | Effective tok/s | Sum abs error | Max abs error |
|---:|---:|---:|---:|
| 1 | 0.301 | 0.709618 | 0.263121 |
| 2 | 0.341 | 0.452253 | 0.188639 |
| 4 | 0.337 | 0.620001 | 0.162198 |
| 8 | 0.333 | 0.497538 | 0.144478 |
| 16 | 0.311 | 0.369923 | 0.122774 |
| 42 / all layers | 0.279 in that run | **0.237515** | **0.093098** |

Conclusion: applying layer scalar after BF16 for all Gemma4 MTP verifier layers remains the best correctness setting. Partial early-layer application regresses selected-logit parity.

### Layer-scalar ordering prompt/verifier isolation

The `BF16(hidden) -> layer_output_scale` improvement was isolated across prompt-context seed rows and verifier rows.

| Scope using scalar-after-BF16 | Effective tok/s | Sum abs error | Max abs error |
|---|---:|---:|---:|
| verifier rows only (`pos >= 2`) | 0.308 | 0.357792 | 0.163836 |
| prompt seed rows only (`pos < 2`) | 0.275 | 0.445313 | 0.213539 |
| all prompt + verifier rows | current best | **0.237515** | **0.093098** |

Logs:

- `logs/mtp-strict-layer-scalar-verifier-only-20260617-012532.log`
- `logs/mtp-strict-layer-scalar-prompt-only-20260617-012541.log`

Conclusion: the improved ordering needs to apply consistently to both prompt-context seed and verifier batch rows in `forwardMTPPromptLayer`. Applying it only to one side regresses selected-logit parity.
