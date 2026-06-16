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
