# Decider native validation — 2026-09-21

This milestone implements the released `Mapika/decider-0.8b` state-first independent scoring path in native Go. Upstream source and weights are Apache-2.0.

## Provenance

- Source: `Mapika/decider` at `c4daaac28af9fea95d627015cffa2dd5a5926ee6`.
- Model: `Mapika/decider-0.8b` at `1ea54127d3bd52f6d753d9257b32a6380b873907`.
- Base: `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Released safetensor SHA-256: `6926f82ef7e9ea408ad881555204daa6bb694ca82509c4d1514b7be2e713563d` (1,504,827,608 bytes).
- Tokenizer SHA-256: `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.
- Model architecture: Qwen3.5, 24 layers, 18 gated-delta linear-attention plus six full-attention layers, hidden size 1,024, tied 248,320×1,024 BF16 embedding/head.
- Released temperature: 1.03; state-first layout; isolated Score levels.

## Numerical evidence

The checked-in CPU-only reference fixture contains exact prompts, token IDs, the complete 255-entry one-token label table, raw candidate logits and typed answers for one Choice, one Noul and one three-level isolated Score request. Transformers 5.17.0 BF16 reference logits are:

| Row | Candidate logits |
|---|---|
| team | `[17.125, 8.125, 8.750]` |
| refund requested | `[9.750, 15.500]` |
| calm fits | `[14.0625, 8.1875]` |
| frustrated fits | `[12.125, 12.3125]` |
| very frustrated fits | `[11.9375, 11.6875]` |

Native Go reproduces every prompt token exactly and every candidate logit within `0.12`. The resulting decisions agree: `billing`, refund probability about `0.996`, and frustration near level `1.4` with `frustrated` most likely. The five-row native run took about 25 seconds and 5.0 GiB peak RSS on the Intel validation host. The Transformers oracle took 10.15 seconds and 2.67 GiB RSS with six CPU threads; these are compatibility measurements, not throughput comparisons.

## Contract and limits

`SystemOne` accepts ordered questions and ordered criteria, renders Python-compatible JSON spacing, inserts wide option labels as explicit known tokens, and reports ordered answers. `Object` is available when JSON property order must match an external fixture. Choice supports 2–255 options, Score supports 2–10 levels, and Noul reports the probability of yes. Long arrays are annotated with `_index` like upstream.

State-first independent scoring is the supported correctness path. Schema-first prefix caching, packed dependent questions, CUDA graphs, FP8, vision, training and RL are not claimed. The production loader admits the pinned 0.8B geometry, strict release sidecar, tied embedding/head and tokenizer label table. A separately named fixture loader permits tiny geometry for bounded tests while retaining the structural checks.

Package statement coverage is **90.0%** without released assets. The tiny synthetic single-choice API takes roughly 16 µs on the Intel host and 58 µs on native ARM64 CIX P1, with 7.3 KB and 301 allocations on both. Profiles attribute the largest allocation sites to tokenizer regex splitting, Qwen full-attention temporaries and answer-map assembly; CPU and heap profiles are retained as validation artifacts.

Final gates passed: 118 race-tested packages plus 54 no-test packages, `go vet ./...`, host build, Linux ARM64 and RISC-V builds, 378-document link/build checks, and native ARM64 package execution. The frozen experiment remains unchanged at 16 manifest entries, two binaries and 676 records.
