# Go System One v1 validation — 21 September 2026

Go System One v1 serves bounded boolean and enum decisions from the pinned Gemma 4 12B GGUF. CPU/SIMD defines the numerical oracle. The NVIDIA path keeps projection weights resident and executes prompt prefill, independent suffix transformers and the LM head through native PTX.

## Frozen inputs

| Input | Pin |
|---|---|
| Model | `unsloth/gemma-4-12b-it-GGUF@fc034cfff751157913579611efad8462ac1be606` |
| File | `gemma-4-12b-it-UD-Q4_K_XL.gguf`, 7,366,423,360 bytes |
| Model SHA-256 | `90fd944d227e9d9b68e7e2c7d5b57b79d4c66ed521b0919fbbd932cf834f6f8e` |
| Tokenizer | `google/gemma-4-12b-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| llama.cpp oracle | `thecodacus/llama.cpp@14d04e755fa28653e87b9a07072892265bdc0fad` |
| Device | NVIDIA GeForce RTX 3060, 12,288 MiB, compute capability 8.6 |
| Driver | 580.173.02 |

`model/gosystemone/testdata/provenance.json` records the model, tokenizer, source and runtime bounds. `go-system-one` verifies all four artifact hashes before parsing them unless `-verify-artifacts=false` is set explicitly.

## Numerical gates

Synthetic differential tests cover Q4_K, Q5_K and Q6_K projection kernels; device-side RoPE; independent GQA; sibling isolation; non-unit layer scaling; shared trunk and branch-local suffix KV; final LM-head logits; cancellation and unsupported PLI rejection.

The opt-in released-model test is:

```sh
GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B=/tmp/go-system-one-gemma4-12b/gemma-4-12b-it-UD-Q4_K_XL.gguf \
GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER=/tmp/go-system-one-gemma4-12b/tokenizer \
go test ./model/gosystemone -run '^TestGoSystemOneNVIDIAReleasedModelMatchesPinnedLlamaCppDecision$' -count=1 -v
```

It uses the checked-in llama.cpp fixture prompt and candidate paths. The native NVIDIA result selected `true` with probability `0.999999999956813`; the pinned oracle probability is `0.999999999905886`. The full-vocabulary one-token diagnostic had the same CPU/SIMD and NVIDIA argmax. Representative logits differed by 0.0035–0.0532 after quantised reduction-order changes.

The second fixture, `llamacpp-go-system-one-gemma4-12b-multifield.json`, adds two contexts, two fields and a three-way multi-token enum. The pinned worker selects `critical incident`/`true` for a production outage and `routine maintenance`/`false` for completed maintenance. Native NVIDIA matches all four winners and probabilities within `1e-6`. The fixture records exact rendered text, SHA-256 values, token IDs, candidate indices and the worker environment.

```sh
GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B=/tmp/go-system-one-gemma4-12b/gemma-4-12b-it-UD-Q4_K_XL.gguf \
GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER=/tmp/go-system-one-gemma4-12b/tokenizer \
go test ./model/gosystemone -run 'TestGoSystemOneNVIDIAMultiFieldReleasedModelMatchesPinnedLlamaCpp' -count=1 -v
```

## Warm decision timing

Startup, artifact hashing, GGUF parsing and the resident upload are excluded. The measured interval starts when the HTTP handler receives `POST /v1/decision` and ends when it writes the complete response. The request used one 77-token prepared prompt, a 20-token context, one boolean tree node and a four-token suffix.

| Path | Result |
|---|---:|
| Native NVIDIA/PTX HTTP decision, 100 warm requests | 79.71–87.99 ms; median 80.65 ms; p95 86.23 ms; p99 86.93 ms |
| Native NVIDIA/PTX first request after startup | 271.7 ms wall time |
| Renamed pre-optimisation warm baseline | 135.3–136.7 ms |
| Earlier native warm baseline | 520.8–521.9 ms |
| Pinned llama.cpp worker prefill | 53.8 ms |
| Pinned llama.cpp worker suffix scoring | 42.2 ms |
| Pinned llama.cpp worker total | 96.0 ms |
| Resident projection bytes | 9,173,277,696 |

The native median is 15.35 ms, or 16.0%, faster than the pinned llama.cpp worker on this fixture. It is 40.9% faster than the renamed 135.3–136.7 ms baseline and 6.5 times faster than the earlier native warm baseline. The Q4_K path uses the pinned revision's Ampere MMQ tile, compiled to embedded PTX at development time. Compatible raw Q4 gate/up weights are concatenated without changing their total resident bytes; one projection feeds the existing fused split/GELU kernel. A byte-exact two-dimensional Q8_1 packer assigns one warp to each `(row, 128-value block)` instead of looping over all blocks in one row. Q4_K matrices stay in canonical GGUF form on the device, which reduces resident projection storage by 1,558,978,560 bytes. Q5_K uses shape-aware 64-output-row J8/J16/J24 tiles. Q6_K uses 64-output-row J8/J12/J16 tiles and a byte-exact grouped 16-value activation packer, selecting J12 or J16 only for measured wide matrices. For the common single-node tree request, the 20 context tokens and four forced suffix tokens execute as one causal 24-row batch, and the LM head projects only the candidate token rows.

A one-token released-model diagnostic measured 121 ms for native NVIDIA scoring and 4.20 s for SIMD scoring after their respective prefilled trunks. CPU/SIMD full-request timing is impractical for interactive use: the earlier 77-token SIMD prefill diagnostic took 565 s. It remains the correctness oracle.

These figures are controlled samples on this host, not a throughput distribution. The native path uses 512-token packed prefill, a shared-prefix device KV snapshot, 12 independent suffix sequences and packed branch scoring. The product runtime has no llama.cpp or CUDA toolkit dependency; the embedded PTX is loaded through the NVIDIA driver API.

## Warp-parallel causal attention

`backends/nvidia/ptx/attention_causal_warp_generated.go` replaces the original serial-dot causal GQA kernel for packed prefill and suffix scoring. The runtime API and launch shape remain unchanged: one CUDA block handles one query head and one query row, with a 256-thread block and one block for every `(head, row)` pair.

The block executes attention in three stages:

1. Eight warps divide the admissible key positions between them. The 32 lanes of each warp split the head dimension and reduce one `Q·K` score with warp shuffles.
2. All 256 threads reduce the softmax maximum and sum. Up to 2,048 scores and probabilities remain in shared memory. Causal and sliding-window bounds are computed from `POS0`, the query-row index, `KV_LEN` and `WINDOW` before scoring.
3. Threads divide the output head dimensions and accumulate the probability-weighted value rows.

The `sm_86` image compiled with CUDA 12.8, `-O3` and `--use_fast_math` uses 61 registers per thread and 9,216 bytes of static shared memory. `ptxas` reported no stack frame or local-memory spills. CUDA is a development-time compiler only; production loads the embedded PTX through the NVIDIA Driver API.

`TestCausalBatchAttentionMatchesCPU` checks full-causal and sliding-window GQA against an independent F32 CPU implementation with a fixed `2e-5` absolute/relative limit. A direct device comparison against the previous PTX measured a maximum absolute difference of `2.98e-8`. Representative kernel timings changed from 47.8 to 19.6 µs for a 24-row sequence starting at position 0, and from 236.1 to 84.6 µs for 24 rows starting at position 77 with 101 physical KV rows.

On the pinned released-model HTTP fixture, the warp-parallel kernel reduced the 48 attention launches from 10.28 ms to 3.84 ms in Nsight Systems. The final profile attributed 80.4 ms to all GPU kernels inside an 84.2 ms profiled HTTP request. Q4 projections used 30.63 ms, Q6 projections 25.40 ms, Q5 projections 15.61 ms, and activation packing 1.36 ms. The released-model parity test passed three consecutive isolated runs and selected `true`; the final 100-request run reported constrained probabilities within `8.7e-11` of the pinned llama.cpp value `0.999999999905886`.

## Retention and robustness

The server completed 100 sequential warm requests after one warm-up. Five clients then cancelled their requests after 10 ms, and 50 malformed requests exercised validation without entering inference. A subsequent valid request returned HTTP 200 with the same `true` decision. Process GPU memory was 9,364 MiB before and after the sequence. Process RSS changed from 12,839,804 KiB to 12,840,948 KiB, an increase of 1,144 KiB. Resident projection accounting stayed at 9,173,277,696 bytes.

The accepted optimisation set has no request-unbounded cache and does not change single admission, cancellation ownership or the 12 GB device budget. Concatenated Q4 gate/up matrices replace the two original resident matrices with the same raw bytes; they do not duplicate weights.

Rejected experiments included Q5 128-output-row tiles, Q6 128/256-output-row tiles outside their measured shapes, Q6 128-thread blocks, 512/1,024-value Q6 K tiles, wider J16/J24 Q6 row tiles, coalesced Q4 gate/up pairs and narrow-shape Q4 J8 dispatch. Each was slower in the released-model shape probe or failed to improve whole-request latency. Numerically incorrect Q5 J16 experiments were discarded before integration.

## Workload matrix and cache admission

Five warm samples per case used the pinned model and RTX 3060. Each schema was warmed before measurement. Times cover the complete HTTP handler.

| Case | Workload | Median | Range |
|---|---|---:|---:|
| Baseline | 20 context tokens, one boolean field, tree, cache hit | 81.00 ms | 80.44–81.94 ms |
| Cache disabled | Baseline with `cache_prompt=false` | 252.06 ms | 251.38–252.70 ms |
| Short context | 15 context tokens, one boolean field | 81.14 ms | 80.90–81.63 ms |
| Long context | 371 context tokens, one boolean field | 1,224.59 ms | 1,221.76–1,231.35 ms |
| Deep enum | Three multi-token candidates, five scored rows | 99.25 ms | 99.05–101.38 ms |
| Four fields | Four booleans, 20 scored rows | 742.57 ms | 739.86–750.00 ms |
| Four contexts | Four contexts, one boolean, four rounds | 327.51 ms | 326.50–328.28 ms |
| Auto tree | Three candidates, `tree_max=128` | 97.84 ms | 97.79–98.41 ms |
| Auto greedy | Three candidates, `tree_max=2` | 97.66 ms | 97.57–98.86 ms |
| Cache miss | Alternate schema, then baseline | 255.03 ms | 252.70–257.63 ms |

The scorer retains one shared schema-prefix context. Its key is the exact shared token sequence, and reuse now also requires enough preallocated trunk capacity for the current context and suffix. A longer context therefore replaces an undersized entry instead of failing with `invalid NVIDIA prefixed branch`. Alternating short contexts under one schema stayed at 80.9–82.7 ms. The one-entry cache is bounded by construction; an LRU for repeated context text would retain substantial per-layer KV and is not justified by this schema-prefix workload.

The handler retains single admission. A second request sent 50 ms into a 1.2-second long-context request received HTTP 429 with `decision inference busy; retry later`. The matrix shows roughly linear context/field multiplication, while the device already uses 9,646 MiB with the larger cached arena. Multi-request batching or concurrent admission would need a separate memory and ownership design; this release keeps the existing policy.

## Product decisions

Go System One v1 keeps boolean and unordered string-enum fields. An ordered-score field needs separate probability-distribution, expected-level and compatibility semantics; no released Go System One checkpoint or labelled task requires it, so v1 does not add one.

The checked-in fixtures establish decision parity, not quality or probability calibration. No unspent labelled calibration/test cohort exists for Go System One. NLL, Brier score, expected calibration error, confident-error and risk/coverage claims therefore remain unavailable. Raw constrained probabilities must be labelled as model probabilities.

The Kev roadmap remains assessment-only. P0 needs a safe head export and independent Transformers fixture; P1/P2 depend on that artifact and explicit implementation approval. No Kev checkpoint is loaded by Go System One, and `/v1/systemone` remains absent.

## HTTP and browser checks

The handler limits requests to 1 MiB, rejects unknown fields and models, serialises inference, propagates cancellation, and accepts 1–256 contexts. Schema compilation accepts boolean and string enum fields only, with at most 32 fields and 255 candidates per field.

The independently authored page is served at `/go-system-one`; it calls only `/v1/decision` and `/go-system-one/v1/status`. Tree-mode API fields include the complete ordered candidate distribution, and the page renders every allowed outcome and constrained model probability while highlighting the selected outcome. Greedy fallback retains its selected-path probability but cannot claim probabilities for unscored complete paths. Chromium checks cover page load, request submission, two contexts, boolean and three-way enum candidate distributions, selected markers, decision rendering, OS-preference-only light/dark theming, mobile fit and the existing embedded chat UI.

```sh
cd webui/frontend
bun x playwright test --config playwright.go.config.ts
```

Three Chromium scenarios passed, including the Go System One page. The final repository gates also passed: `go test ./...`, `go vet ./...`, `go test -race ./model/gosystemone ./webui ./cmd/llm/go-system-one`, `git diff --check`, the released-model NVIDIA/llama.cpp parity test above, and compile-only `linux/arm64` and `linux/riscv64` builds of `go-system-one`, `model/gosystemone` and `webui`.

## Native and compile-only scope

The NVIDIA tests ran natively on the RTX 3060. `linux/arm64` and `linux/riscv64` checks are compile-only and must not be described as execution on those architectures.
