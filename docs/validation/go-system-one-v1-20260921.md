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

## Warm decision timing

Startup, artifact hashing, GGUF parsing and the resident upload are excluded. The measured interval starts when the HTTP handler receives `POST /v1/decision` and ends when it writes the complete response. The request used one 77-token prepared prompt, a 20-token context, one boolean tree node and a four-token suffix.

| Path | Result |
|---|---:|
| Native NVIDIA/PTX HTTP decision, six warm requests | 135.3–136.7 ms |
| Native NVIDIA/PTX first request after startup | 410.3 ms |
| Earlier native warm baseline | 520.8–521.9 ms |
| Pinned llama.cpp worker prefill | 53.8 ms |
| Pinned llama.cpp worker suffix scoring | 42.2 ms |
| Pinned llama.cpp worker total | 96.0 ms |
| Resident projection bytes | 9,173,277,696 |

The native result is 1.41 times slower than the pinned llama.cpp worker on this fixture and 3.8 times faster than the earlier native warm baseline. The Q4_K path uses the pinned revision's Ampere MMQ tile, compiled to embedded PTX at development time. Q4_K matrices stay in canonical GGUF form on the device, which reduces resident projection storage by 1,558,978,560 bytes. Q5_K and Q6_K retain their upload-time packed representations. For the common single-node tree request, the 20 context tokens and four forced suffix tokens execute as one causal 24-row batch, and the LM head projects only the candidate token rows.

A one-token released-model diagnostic measured 121 ms for native NVIDIA scoring and 4.20 s for SIMD scoring after their respective prefilled trunks. CPU/SIMD full-request timing is impractical for interactive use: the earlier 77-token SIMD prefill diagnostic took 565 s. It remains the correctness oracle.

These figures are controlled samples on this host, not a throughput distribution. The native path uses 512-token packed prefill, a shared-prefix device KV snapshot, 12 independent suffix sequences and packed branch scoring. The product runtime has no llama.cpp or CUDA toolkit dependency; the embedded PTX is loaded through the NVIDIA driver API.

## HTTP and browser checks

The handler limits requests to 1 MiB, rejects unknown fields and models, serialises inference, propagates cancellation, and accepts 1–256 contexts. Schema compilation accepts boolean and string enum fields only, with at most 32 fields and 255 candidates per field.

The independently authored page is served at `/go-system-one`; it calls only `/v1/decision` and `/go-system-one/v1/status`. Chromium checks cover page load, request submission, decision rendering, constrained probability rendering and the existing embedded chat UI.

```sh
cd webui/frontend
bun x playwright test --config playwright.go.config.ts
```

Three Chromium scenarios passed, including the Go System One page. The final repository gates also passed: `go test ./...`, `go vet ./...`, `go test -race ./model/gosystemone ./webui ./cmd/llm/go-system-one`, `git diff --check`, the released-model NVIDIA/llama.cpp parity test above, and compile-only `linux/arm64` and `linux/riscv64` builds of `go-system-one`, `model/gosystemone` and `webui`.

## Native and compile-only scope

The NVIDIA tests ran natively on the RTX 3060. `linux/arm64` and `linux/riscv64` checks are compile-only and must not be described as execution on those architectures.
